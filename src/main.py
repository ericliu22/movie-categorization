import argparse

def main() -> None:
    """
    Main entry point for the application.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Run the movie genre classification app.")
    parser.add_argument("command", type=str, choices=["train","predict"], help="Path to the input file.")
    parser.add_argument("--output", type=str, help="Path to save the output.")
    args = parser.parse_args()

    command: str = args.command
    output_path: str = args.output

    if command == "train":
        print(f"Training the model")
    elif command == "predict":
        print(f"Predicting movie genre")
    
    if output_path:
        print(f"Output will be saved to: {output_path}")

    message: str = "Hello, world! Welcome to my project."

    print(message)

if __name__ == "__main__":
    main()
