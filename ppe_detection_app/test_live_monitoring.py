"""
Test script to verify live monitoring with integrated webcam
Run this while the Flask app is running
"""
import cv2
import requests
import time
import sys

def test_camera_direct():
    """Test camera directly with OpenCV"""
    print("=" * 60)
    print("Test 1: Direct Camera Access")
    print("=" * 60)
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("❌ FAILED: Could not open camera")
        return False
    
    print("✅ Camera opened successfully")
    
    # Try to read a few frames
    for i in range(3):
        ret, frame = camera.read()
        if ret:
            print(f"✅ Frame {i+1} read successfully (shape: {frame.shape})")
        else:
            print(f"❌ FAILED: Could not read frame {i+1}")
            camera.release()
            return False
        time.sleep(0.1)
    
    camera.release()
    print("✅ Direct camera access test PASSED\n")
    return True

def test_video_feed_endpoint():
    """Test video feed endpoint"""
    print("=" * 60)
    print("Test 2: Video Feed Endpoint")
    print("=" * 60)
    
    try:
        url = 'http://localhost:5000/video_feed'
        print(f"Connecting to {url}...")
        
        response = requests.get(url, stream=True, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Video feed endpoint accessible")
            print(f"   Status Code: {response.status_code}")
            print(f"   Content-Type: {response.headers.get('Content-Type')}")
            
            # Try to read a chunk of the stream
            chunk = next(response.iter_content(chunk_size=1024), None)
            if chunk:
                print(f"✅ Received data chunk ({len(chunk)} bytes)")
                print("✅ Video feed endpoint test PASSED\n")
                return True
            else:
                print("❌ FAILED: No data received from stream")
                return False
        else:
            print(f"❌ FAILED: Status code {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Could not connect to server")
        print("   Make sure the Flask app is running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_detection_status():
    """Test detection status endpoint"""
    print("=" * 60)
    print("Test 3: Detection Status Endpoint")
    print("=" * 60)
    
    try:
        url = 'http://localhost:5000/detection_status'
        print(f"Connecting to {url}...")
        
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Detection status endpoint accessible")
            print(f"   Model Loaded: {data.get('model_loaded', 'N/A')}")
            print(f"   Camera Active: {data.get('camera_active', 'N/A')}")
            print(f"   Camera Available: {data.get('camera_available', 'N/A')}")
            print(f"   Recent Violations: {data.get('recent_violations', 'N/A')}")
            print("✅ Detection status test PASSED\n")
            return True
        else:
            print(f"❌ FAILED: Status code {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Could not connect to server")
        print("   Make sure the Flask app is running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PPE Detection - Live Monitoring Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Direct camera access
    results.append(("Direct Camera Access", test_camera_direct()))
    
    # Test 2: Video feed endpoint (requires Flask app running)
    print("NOTE: The following tests require the Flask app to be running.")
    print("Start the app with: python app.py\n")
    input("Press Enter when the Flask app is running, or Ctrl+C to skip endpoint tests...")
    
    results.append(("Video Feed Endpoint", test_video_feed_endpoint()))
    results.append(("Detection Status", test_detection_status()))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Live monitoring should be working.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

