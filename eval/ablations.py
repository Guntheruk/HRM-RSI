import argparse


def main():
    parser = argparse.ArgumentParser(description="Run RSI ablations.")
    parser.add_argument("--no-vane", action="store_true", help="Remove drift sensing")
    parser.add_argument("--no-s3star", action="store_true", help="Disable interrupts")
    parser.add_argument("--fixed-budget", action="store_true", help="Disable adaptive interrupts")
    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    main()
