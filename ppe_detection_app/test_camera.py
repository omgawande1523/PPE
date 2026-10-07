"""
Test script to verify camera and video feed functionality
"""
import cv2
import requests
import time

def test_camera():
    """Test if camera can be opened and read"""
    print("Testing camera access...")
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("❌ ERROR: Could not open camera")
        return False
    
    print("✅ Camera opened successfully")
    
    # Try to read a frame
    ret, frame = camera.read()
    if not ret:
        print("❌ ERROR: Could not read frame from camera")
        camera.release()
        return False
    
    print(f"✅ Frame read successfully (shape: {frame.shape})")
    camera.release()
    return True

def test_video_feed():
    """Test if video feed endpoint is accessible"""
    print("\nTesting video feed endpoint...")
    try:
        # Try to connect to the video feed
        response = requests.get('http://localhost:5000/video_feed', stream=True, timeout=5)
        if response.status_code == 200:
            print("✅ Video feed endpoint is accessible")
            print(f"   Content-Type: {response.headers.get('Content-Type')}")
            return True
        else:
            print(f"❌ ERROR: Video feed returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to server. Is the Flask app running?")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("PPE Detection - Camera Test")
    print("=" * 50)
    
    camera_ok = test_camera()
    feed_ok = test_video_feed()
    
    print("\n" + "=" * 50)
    if camera_ok and feed_ok:
        print("✅ All tests passed! Camera and video feed are working.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 50)

