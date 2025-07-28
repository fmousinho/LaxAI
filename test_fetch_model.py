from core.common.detection import DetectionModel

def test_fetch_model():
    try:
        detection_model = DetectionModel()
        print("Model fetched and loaded successfully.")
    except RuntimeError as e:
        print(f"Failed to fetch or load the model: {e}")

if __name__ == "__main__":
    test_fetch_model()
