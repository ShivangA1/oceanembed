import copernicusmarine

PRODUCT_ID = "GLOBAL_MULTIYEAR_PHY_001_030"
DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"


def main():
    print("Testing Copernicus Marine connection...\n")

    try:
        catalogue = copernicusmarine.describe(
            product_id=PRODUCT_ID,
            disable_progress_bar=False,
        )

        print("\nSUCCESS!")
        print(f"Product: {PRODUCT_ID}\n")

        for product in catalogue.products:
            print(f"Product name: {product.title}")

            print("\nAvailable datasets:")
            for dataset in product.datasets:
                print(f"  {dataset.dataset_id}")

    except Exception as exc:
        print("\nFAILED")
        print(exc)


if __name__ == "__main__":
    main()