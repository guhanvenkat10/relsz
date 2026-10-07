if __name__ == "__main__":
    import argparse

    from relsz.main import main

    parser = argparse.ArgumentParser(description="Seizure detection on an EDF file; writes a SzCORE annotation TSV.")
    parser.add_argument("input", help="Path to the input EDF EEG file.")
    parser.add_argument("output", help="Path to the output TSV annotation file.")
    a = parser.parse_args()
    main(a.input, a.output)
