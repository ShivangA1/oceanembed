import React, {useEffect, useMemo, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './styles.css';

const API = import.meta.env.VITE_AI_URL || 'http://localhost:8000';

function MapPanel({lat, lon, onPick}) {
  const el = useRef(null);
  const map = useRef(null);
  const marker = useRef(null);

  useEffect(() => {
    map.current = L.map(el.current, {
      minZoom: 4,
      maxZoom: 9,
      maxBounds: [[5, 45], [30, 105]],
      zoomControl: false,
      attributionControl: true,
    }).setView([lat, lon], 5);

    L.control.zoom({position: 'bottomright'}).addTo(map.current);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
    }).addTo(map.current);

    marker.current = L.circleMarker([lat, lon], {
      radius: 7,
      weight: 2,
      color: '#e6fffb',
      fillColor: '#28c7b8',
      fillOpacity: 1,
    }).addTo(map.current);

    map.current.on('click', e => {
      if (e.latlng.lat < 5 || e.latlng.lat > 30 || e.latlng.lng < 45 || e.latlng.lng > 105) return;
      marker.current.setLatLng(e.latlng);
      onPick(e.latlng.lat, e.latlng.lng);
    });

    return () => map.current?.remove();
  }, []);

  useEffect(() => {
    if (marker.current) marker.current.setLatLng([lat, lon]);
    if (map.current) map.current.panTo([lat, lon], {animate: true, duration: 0.35});
  }, [lat, lon]);

  return (
    <div className="map-wrap">
      <div ref={el} className="map" />
      <div className="map-bounds">5–30°N&nbsp;&nbsp;·&nbsp;&nbsp;45–105°E</div>
    </div>
  );
}

function ProfileChart({data}) {
  if (!data) {
    return (
      <div className="empty-state">
        <div className="empty-mark">⌁</div>
        <p>Select a point and reconstruct its temperature profile.</p>
      </div>
    );
  }

  const min = Math.min(...data.temperatures);
  const max = Math.max(...data.temperatures);
  const range = Math.max(0.1, max - min);
  const points = data.temperatures
    .map((t, i) => `${44 + ((t - min) / range) * 242},${18 + (data.depths[i] / 1000) * 365}`)
    .join(' ');

  return (
    <div>
      <div className="chart-head">
        <div>
          <div className="section-kicker">PROFILE</div>
          <h3>Temperature by depth</h3>
          <p>{data.lat.toFixed(2)}°N, {data.lon.toFixed(2)}°E <span>·</span> {data.date}</p>
        </div>
        <span className="mode-pill">{data.mode}</span>
      </div>

      <svg className="profile" viewBox="0 0 330 415" preserveAspectRatio="none">
        <defs>
          <linearGradient id="profileFill" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="#39d7c6" stopOpacity="0.05" />
            <stop offset="100%" stopColor="#39d7c6" stopOpacity="0.18" />
          </linearGradient>
        </defs>
        {[0, 250, 500, 750, 1000].map(d => (
          <g key={d}>
            <line x1="44" x2="310" y1={18 + d * 0.365} y2={18 + d * 0.365} className="grid" />
            <text x="4" y={23 + d * 0.365} className="tick">{d}m</text>
          </g>
        ))}
        <line x1="44" y1="18" x2="44" y2="383" className="axis" />
        <polygon points={`44,18 ${points} 44,383`} fill="url(#profileFill)" />
        <polyline points={points} className="profile-line" />
        {data.temperatures.map((t, i) => {
          const x = 44 + ((t - min) / range) * 242;
          const y = 18 + (data.depths[i] / 1000) * 365;
          return <circle key={i} cx={x} cy={y} r="3.4" className="dot" />;
        })}
      </svg>

      <div className="profile-summary">
        <div><span>Surface</span><strong>{data.temperatures[0].toFixed(2)} °C</strong></div>
        <div><span>Deepest</span><strong>{data.temperatures.at(-1).toFixed(2)} °C</strong></div>
        <div><span>Levels</span><strong>{data.depths.length}</strong></div>
      </div>

      <div className="table">
        {data.depths.map((d, i) => (
          <div className="row" key={d}>
            <span>{d} m</span>
            <b>{data.temperatures[i].toFixed(2)} °C</b>
          </div>
        ))}
      </div>
    </div>
  );
}

function Embedding({values}) {
  if (!values?.length) {
    return <div className="embedding-empty">Embedding values will appear after reconstruction.</div>;
  }

  const max = Math.max(...values.map(v => Math.abs(v)), 1e-6);

  return (
    <div className="embedding-visual">
      <div className="embedding-grid">
        {values.map((v, i) => (
          <div
            key={i}
            title={`Dimension ${i + 1}: ${v}`}
            className="cell"
            style={{opacity: 0.2 + 0.8 * Math.abs(v) / max}}
          >
            {v.toFixed(2)}
          </div>
        ))}
      </div>
      <div className="embedding-meta">
        <span>64 dimensions</span>
        <span>CNN latent representation</span>
      </div>
    </div>
  );
}

function App() {
  const [lat, setLat] = useState(17.5);
  const [lon, setLon] = useState(75);
  const [date, setDate] = useState('2019-01-03');
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState('Checking service');

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(x => setHealth(x.mode === 'trained-model' ? 'Model online' : 'Demo model'))
      .catch(() => setHealth('Service offline'));
  }, []);

  const predict = async () => {
    setBusy(true);
    setError('');
    try {
      const r = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({lat: +lat, lon: +lon, date}),
      });
      const x = await r.json();
      if (!r.ok) throw new Error(x.detail || x.error || 'Prediction failed');
      setResult(x);
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  const pick = (a, b) => {
    setLat(+a.toFixed(3));
    setLon(+b.toFixed(3));
  };

  const surface = useMemo(() => result?.temperatures?.[0], [result]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="logo"><span /></div>
          <div>
            <h1>OceanEmbed</h1>
            <p>Subsurface ocean temperature reconstruction</p>
          </div>
        </div>
        <div className="service-status">
          <span className={`status-dot ${health === 'Service offline' ? 'offline' : ''}`} />
          {health}
        </div>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <div className="eyebrow">NORTH INDIAN OCEAN</div>
          <h2>Explore what lies<br /><span>beneath the surface.</span></h2>
          <p>Choose a location and date to reconstruct the ocean temperature profile from surface observations.</p>
        </div>

        <div className="stats">
          <div><strong>15</strong><span>depth levels</span></div>
          <div><strong>64</strong><span>embedding dims</span></div>
          <div><strong>0.25°</strong><span>grid resolution</span></div>
        </div>
      </section>

      <section className="query-bar card">
        <div className="field">
          <label>Latitude</label>
          <div className="input-wrap"><input type="number" min="5" max="30" step="0.25" value={lat} onChange={e => setLat(e.target.value)} /><span>°N</span></div>
        </div>
        <div className="field">
          <label>Longitude</label>
          <div className="input-wrap"><input type="number" min="45" max="105" step="0.25" value={lon} onChange={e => setLon(e.target.value)} /><span>°E</span></div>
        </div>
        <div className="field date-field">
          <label>Date</label>
          <input type="date" value={date} onChange={e => setDate(e.target.value)} />
        </div>
        <button onClick={predict} disabled={busy}>
          {busy ? <><span className="spinner" />Reconstructing</> : <>Reconstruct <span>→</span></>}
        </button>
      </section>

      {error && <div className="error card">{error}</div>}

      <main className="dashboard">
        <section className="card map-card">
          <div className="panel-title">
            <div>
              <div className="section-kicker">LOCATION</div>
              <h3>Ocean map</h3>
            </div>
            <span className="coord">{(+lat).toFixed(2)}°N&nbsp; / &nbsp;{(+lon).toFixed(2)}°E</span>
          </div>
          <MapPanel lat={+lat} lon={+lon} onPick={pick} />
        </section>

        <section className="card profile-card">
          <ProfileChart data={result} />
        </section>
      </main>

      <section className="card embedding-card">
        <div className="panel-title">
          <div>
            <div className="section-kicker">REPRESENTATION</div>
            <h3>Ocean embedding</h3>
          </div>
          {surface != null && <span className="surface">Surface&nbsp; {surface.toFixed(2)} °C</span>}
        </div>
        <Embedding values={result?.embedding} />
      </section>

      <footer>OceanEmbed</footer>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
