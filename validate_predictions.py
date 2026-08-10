import argparse
import json
import sys
from jsonschema import validate, ValidationError

PREDICTIONS_SCHEMA_PATH = "schemas/predictions.schema.json"

def main():
    parser = argparse.ArgumentParser(description="Validate predictions.json structure and manifest coverage.")
    parser.add_argument("predictions", help="Path to predictions.json")
    parser.add_argument("--manifest", required=True, help="Path to input manifest.json")
    args = parser.parse_args()

    # Load predictions JSON
    try:
        with open(args.predictions, "r") as f:
            predictions_data = json.load(f)
    except Exception as e:
        print(f"FAILED: Unable to load predictions file: {e}")
        sys.exit(1)

    # Load schema
    try:
        with open(PREDICTIONS_SCHEMA_PATH, "r") as f:
            schema_data = json.load(f)
    except Exception as e:
        print(f"FAILED: Unable to load schema file at {PREDICTIONS_SCHEMA_PATH}: {e}")
        sys.exit(1)

    # Load input manifest
    try:
        with open(args.manifest, "r") as f:
            manifest_data = json.load(f)
    except Exception as e:
        print(f"FAILED: Unable to load manifest file: {e}")
        sys.exit(1)

    # 1. Validate JSON schema
    try:
        validate(instance=predictions_data, schema=schema_data)
        print("PASS: Schema validation successful.")
    except ValidationError as e:
        print(f"FAILED: JSON Schema validation error: {e.message}")
        sys.exit(1)

    # 2. Validate scene coverage against manifest
    manifest_scenes = [s["scene_id"] for s in manifest_data.get("scenes", [])]
    prediction_scenes = [s["scene_id"] for s in predictions_data.get("scenes", [])]

    if set(manifest_scenes) != set(prediction_scenes):
        print(f"FAILED: Scene mismatch!")
        print(f"Expected scenes in manifest: {manifest_scenes}")
        print(f"Found scenes in predictions: {prediction_scenes}")
        sys.exit(1)

    if len(manifest_scenes) != len(prediction_scenes):
        print(f"FAILED: Scene count mismatch! Duplicates present.")
        sys.exit(1)

    print("PASS: Scene coverage matches input manifest exactly.")
    print("ALL CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()