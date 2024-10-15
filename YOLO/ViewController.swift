//  Blue Horus - AGPL-3.0 License
//
// Main View Controller for Blue Horus App
// This file is a component of the Blue Horus app, which facilitates real-time object detection using YOLOv8 models on iOS devices.
// Licensed under AGPL-3.0.

// This ViewController oversees the app's primary interface, managing video capture, selection of detection models,
// visualization of detection results, and user interactions. It configures and maintains the video preview layer, 
// enables model switching with a segmented control, handles UI elements such as sliders for adjusting confidence 
// and IoU thresholds, and overlays detection results on the video stream.
// The implementation utilizes the CoreML, Vision, and AVFoundation frameworks to execute real-time object detection 
// and manage interactions with the device's camera hardware.


import AVFoundation
import CoreML
import CoreMedia
import UIKit
import Vision

var mlModel = try! yolov8m(configuration: .init()).model

class ViewController: UIViewController {
  @IBOutlet var videoPreview: UIView!
  @IBOutlet var View0: UIView!
  @IBOutlet var segmentedControl: UISegmentedControl!
  @IBOutlet var playButtonOutlet: UIBarButtonItem!
  @IBOutlet var pauseButtonOutlet: UIBarButtonItem!
  @IBOutlet var slider: UISlider!
  @IBOutlet var sliderConf: UISlider!
  @IBOutlet weak var sliderConfLandScape: UISlider!
  @IBOutlet var sliderIoU: UISlider!
  @IBOutlet weak var sliderIoULandScape: UISlider!
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  @IBOutlet weak var labelZoom: UILabel!
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var labelSlider: UILabel!
  @IBOutlet weak var labelSliderConf: UILabel!
  @IBOutlet weak var labelSliderConfLandScape: UILabel!
  @IBOutlet weak var labelSliderIoU: UILabel!
  @IBOutlet weak var labelSliderIoULandScape: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var forcus: UIImageView!
  @IBOutlet weak var toolBar: UIToolbar!

  let selection = UISelectionFeedbackGenerator()
  var detector = try! VNCoreMLModel(for: mlModel)
  var session: AVCaptureSession!
  var videoCapture: VideoCapture!
  var currentBuffer: CVPixelBuffer?
  var framesDone = 0
  var t0 = 0.0  // inference start
  var t1 = 0.0  // inference dt
  var t2 = 0.0  // inference dt smoothed
  var t3 = CACurrentMediaTime()  // FPS start
  var t4 = 0.0  // FPS dt smoothed
  // var cameraOutput: AVCapturePhotoOutput!

  // Developer mode
  let developerMode = UserDefaults.standard.bool(forKey: "developer_mode")  // developer mode selected in settings
  let save_detections = false  // write every detection to detections.txt
  let save_frames = false  // write every frame to frames.txt

  lazy var visionRequest: VNCoreMLRequest = {
    let request = VNCoreMLRequest(
      model: detector,
      completionHandler: {
        [weak self] request, error in
        self?.processObservations(for: request, error: error)
      })
    // NOTE: BoundingBoxView object scaling depends on request.imageCropAndScaleOption https://developer.apple.com/documentation/vision/vnimagecropandscaleoption
    request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
    return request
  }()

  override func viewDidLoad() {
    super.viewDidLoad()
    
    slider.value = 30
    setUpBoundingBoxViews()
    setUpOrientationChangeNotification()
    startVideo()
  }

  override func viewWillTransition(
    to size: CGSize, with coordinator: any UIViewControllerTransitionCoordinator
  ) {
    super.viewWillTransition(to: size, with: coordinator)

    if size.width > size.height {
      labelSliderConf.isHidden = true
      sliderConf.isHidden = true
      labelSliderIoU.isHidden = true
      sliderIoU.isHidden = true
      toolBar.setBackgroundImage(UIImage(), forToolbarPosition: .any, barMetrics: .default)
      toolBar.setShadowImage(UIImage(), forToolbarPosition: .any)

      labelSliderConfLandScape.isHidden = false
      sliderConfLandScape.isHidden = false
      labelSliderIoULandScape.isHidden = false
      sliderIoULandScape.isHidden = false

    } else {
      labelSliderConf.isHidden = false
      sliderConf.isHidden = false
      labelSliderIoU.isHidden = false
      sliderIoU.isHidden = false
      toolBar.setBackgroundImage(nil, forToolbarPosition: .any, barMetrics: .default)
      toolBar.setShadowImage(nil, forToolbarPosition: .any)

      labelSliderConfLandScape.isHidden = true
      sliderConfLandScape.isHidden = true
      labelSliderIoULandScape.isHidden = true
      sliderIoULandScape.isHidden = true
    }
    self.videoCapture.previewLayer?.frame = CGRect(
      x: 0, y: 0, width: size.width, height: size.height)

  }

  private func setUpOrientationChangeNotification() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(orientationDidChange),
      name: UIDevice.orientationDidChangeNotification, object: nil)
  }

  @objc func orientationDidChange() {
    videoCapture.updateVideoOrientation()
  }

  @IBAction func vibrate(_ sender: Any) {
    selection.selectionChanged()
  }

    @IBAction func indexChanged(_ sender: UISegmentedControl) {
        selection.selectionChanged()
        activityIndicator.startAnimating()

        switch segmentedControl.selectedSegmentIndex {
        case 0:
            // Remove or comment out the following line:
            // self.labelName.text = "yolov8n"
            do {
                mlModel = try yolov8n(configuration: .init()).model
            } catch {
                print("Error loading model: \(error)")
            }
        case 1:
            // Remove or comment out the following line:
            // self.labelName.text = "yolov8s"
            do {
                mlModel = try bestSmall(configuration: .init()).model
            } catch {
                print("Error loading model: \(error)")
            }
        default:
            break
        }

        setModel()
        setUpBoundingBoxViews()
        activityIndicator.stopAnimating()
    }

  func setModel() {

    /// VNCoreMLModel
    detector = try! VNCoreMLModel(for: mlModel)
    detector.featureProvider = ThresholdProvider()

    /// VNCoreMLRequest
    let request = VNCoreMLRequest(
      model: detector,
      completionHandler: { [weak self] request, error in
        self?.processObservations(for: request, error: error)
      })
    request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
    visionRequest = request
    t2 = 0.0  // inference dt smoothed
    t3 = CACurrentMediaTime()  // FPS start
    t4 = 0.0  // FPS dt smoothed
  }

  /// Update thresholds from slider values
  @IBAction func sliderChanged(_ sender: Any) {
    let conf = Double(round(100 * sliderConf.value)) / 100
    let iou = Double(round(100 * sliderIoU.value)) / 100
    self.labelSliderConf.text = String(conf) + " Confidence Threshold"
    self.labelSliderIoU.text = String(iou) + " IoU Threshold"
    detector.featureProvider = ThresholdProvider(iouThreshold: iou, confidenceThreshold: conf)
  }

  @IBAction func playButton(_ sender: Any) {
    selection.selectionChanged()
    self.videoCapture.start()
    playButtonOutlet.isEnabled = false
    pauseButtonOutlet.isEnabled = true
  }

  @IBAction func pauseButton(_ sender: Any?) {
    selection.selectionChanged()
    self.videoCapture.stop()
    playButtonOutlet.isEnabled = true
    pauseButtonOutlet.isEnabled = false
  }

  // share image
  @IBAction func shareButton(_ sender: Any) {
    selection.selectionChanged()
    let settings = AVCapturePhotoSettings()
    self.videoCapture.cameraOutput.capturePhoto(
      with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
  }

  let maxBoundingBoxViews = 100
  var boundingBoxViews = [BoundingBoxView]()
  var colors: [String: UIColor] = [:]

  func setUpBoundingBoxViews() {
    // Ensure all bounding box views are initialized up to the maximum allowed.
    while boundingBoxViews.count < maxBoundingBoxViews {
      boundingBoxViews.append(BoundingBoxView())
    }

    // Retrieve class labels directly from the CoreML model's class labels, if available.
    guard let classLabels = mlModel.modelDescription.classLabels as? [String] else {
      fatalError("Class labels are missing from the model description")
    }

    // Assign random colors to the classes.
    for label in classLabels {
            if colors[label] == nil {  // if key not in dict
                colors[label] = UIColor(
                    red: CGFloat.random(in: 0...1),
                    green: CGFloat.random(in: 0...1),
                    blue: CGFloat.random(in: 0...1),
                    alpha: 0.6)
            }
    }
  }

  func startVideo() {
    videoCapture = VideoCapture()
    videoCapture.delegate = self

    videoCapture.setUp(sessionPreset: .photo) { success in
      // .hd4K3840x2160 or .photo (4032x3024)  Warning: 4k may not work on all devices i.e. 2019 iPod
      if success {
        // Add the video preview into the UI.
        if let previewLayer = self.videoCapture.previewLayer {
          self.videoPreview.layer.addSublayer(previewLayer)
          self.videoCapture.previewLayer?.frame = self.videoPreview.bounds  // resize preview layer
        }

        // Add the bounding box layers to the UI, on top of the video preview.
        for box in self.boundingBoxViews {
          box.addToLayer(self.videoPreview.layer)
        }

        // Once everything is set up, we can start capturing live video.
        self.videoCapture.start()
      }
    }
  }

  func predict(sampleBuffer: CMSampleBuffer) {
    if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
      currentBuffer = pixelBuffer

      /// - Tag: MappingOrientation
      // The frame is always oriented based on the camera sensor,
      // so in most cases Vision needs to rotate it for the model to work as expected.
      let imageOrientation: CGImagePropertyOrientation
      switch UIDevice.current.orientation {
      case .portrait:
        imageOrientation = .up
      case .portraitUpsideDown:
        imageOrientation = .down
      case .landscapeLeft:
        imageOrientation = .up
      case .landscapeRight:
        imageOrientation = .up
      case .unknown:
        imageOrientation = .up
      default:
        imageOrientation = .up
      }

      // Invoke a VNRequestHandler with that image
      let handler = VNImageRequestHandler(
        cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
      if UIDevice.current.orientation != .faceUp {  // stop if placed down on a table
        t0 = CACurrentMediaTime()  // inference start
        do {
          try handler.perform([visionRequest])
        } catch {
          print(error)
        }
        t1 = CACurrentMediaTime() - t0  // inference dt
      }

      currentBuffer = nil
    }
  }

  func processObservations(for request: VNRequest, error: Error?) {
    DispatchQueue.main.async {
      if let results = request.results as? [VNRecognizedObjectObservation] {
        self.show(predictions: results)
      } else {
        self.show(predictions: [])
      }

      // Measure FPS
      if self.t1 < 10.0 {  // valid dt
        self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
      }
      self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
      self.t3 = CACurrentMediaTime()
    }
  }

  // Return hard drive space (GB)
  func freeSpace() -> Double {
    let fileURL = URL(fileURLWithPath: NSHomeDirectory() as String)
    do {
      let values = try fileURL.resourceValues(forKeys: [
        .volumeAvailableCapacityForImportantUsageKey
      ])
      return Double(values.volumeAvailableCapacityForImportantUsage!) / 1E9  // Bytes to GB
    } catch {
      print("Error retrieving storage capacity: \(error.localizedDescription)")
    }
    return 0
  }


    func show(predictions: [VNRecognizedObjectObservation]) {
        let width = videoPreview.bounds.width  // e.g., 375 pixels
        let height = videoPreview.bounds.height  // e.g., 812 pixels
        var str = ""

        // Calculate the aspect ratio difference between the video preview and the capture session.
        var ratio: CGFloat = 1.0
        if videoCapture.captureSession.sessionPreset == .photo {
            ratio = (height / width) / (4.0 / 3.0)  // For .photo preset
        } else {
            ratio = (height / width) / (16.0 / 9.0)  // For other presets like .hd4K3840x2160, .hd1920x1080, etc.
        }

        // Get the current time in seconds since midnight for logging purposes.
        let date = Date()
        let calendar = Calendar.current
        let hour = calendar.component(.hour, from: date)
        let minutes = calendar.component(.minute, from: date)
        let seconds = calendar.component(.second, from: date)
        let nanoseconds = calendar.component(.nanosecond, from: date)
        let sec_day = Double(hour) * 3600.0 + Double(minutes) * 60.0 + Double(seconds) + Double(nanoseconds) / 1E9  // Seconds in the day

        // **Filter predictions to only include those labeled 'person'**
        let personPredictions = predictions.filter { prediction in
            if let bestLabel = prediction.labels.first {
                return bestLabel.identifier == "person"
            }
            return false
        }

        // Update the label with the count of 'person' predictions
        self.labelSlider.text = "\(personPredictions.count) items (max \(Int(slider.value)))"

        // Loop through the bounding box views
        for i in 0..<boundingBoxViews.count {
            if i < personPredictions.count && i < Int(slider.value) {
                let prediction = personPredictions[i]

                var rect = prediction.boundingBox  // Normalized coordinates (origin at bottom-left)
                switch UIDevice.current.orientation {
                case .portraitUpsideDown:
                    rect = CGRect(
                        x: 1.0 - rect.origin.x - rect.width,
                        y: 1.0 - rect.origin.y - rect.height,
                        width: rect.width,
                        height: rect.height)
                case .landscapeLeft, .landscapeRight, .portrait, .faceUp, .faceDown:
                    // No change needed for these orientations
                    break
                case .unknown:
                    print("The device orientation is unknown; the predictions may be affected.")
                    fallthrough
                default:
                    break
                }

                if ratio >= 1 {  // For iPhones (ratio > 1)
                    let offset = (1 - ratio) * (0.5 - rect.minX)
                    let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
                    rect = rect.applying(transform)
                    rect.size.width *= ratio
                } else {  // For iPads (ratio < 1)
                    let offset = (ratio - 1) * (0.5 - rect.maxY)
                    let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
                    rect = rect.applying(transform)
                    ratio = (height / width) / (3.0 / 4.0)
                    rect.size.height /= ratio
                }

                // Convert normalized coordinates to pixel coordinates
                rect = VNImageRectForNormalizedRect(rect, Int(width), Int(height))

                // Get the best class label and confidence
                let bestClass = prediction.labels[0].identifier
                let confidence = prediction.labels[0].confidence
                // print(confidence, rect)  // Debugging output if needed

                let label = String(format: "%@ %.1f", bestClass, confidence * 100)
                let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)  // Adjust alpha based on confidence

                // Show the bounding box
                boundingBoxViews[i].show(
                    frame: rect,
                    label: label,
                    color: colors[bestClass] ?? UIColor.white,
                    alpha: alpha)  // Alpha ranges from 0 (transparent) to 1 (opaque)

                if developerMode {
                    // Write detection data to a string (or file if implemented)
                    if save_detections {
                        str += String(format: "%.3f %.3f %.3f %@ %.2f %.1f %.1f %.1f %.1f\n",
                                      sec_day, freeSpace(), UIDevice.current.batteryLevel, bestClass, confidence,
                                      rect.origin.x, rect.origin.y, rect.size.width, rect.size.height)
                    }
                }
            } else {
                // Hide any unused bounding box views
                boundingBoxViews[i].hide()
            }
        }

        // If needed, hide any remaining bounding boxes beyond the current count
        for i in personPredictions.count..<boundingBoxViews.count {
            boundingBoxViews[i].hide()
        }
    }


  // Pinch to Zoom Start ---------------------------------------------------------------------------------------------
  let minimumZoom: CGFloat = 1.0
  let maximumZoom: CGFloat = 10.0
  var lastZoomFactor: CGFloat = 1.0

  @IBAction func pinch(_ pinch: UIPinchGestureRecognizer) {
    let device = videoCapture.captureDevice

    // Return zoom value between the minimum and maximum zoom values
    func minMaxZoom(_ factor: CGFloat) -> CGFloat {
      return min(min(max(factor, minimumZoom), maximumZoom), device.activeFormat.videoMaxZoomFactor)
    }

    func update(scale factor: CGFloat) {
      do {
        try device.lockForConfiguration()
        defer {
          device.unlockForConfiguration()
        }
        device.videoZoomFactor = factor
      } catch {
        print("\(error.localizedDescription)")
      }
    }

    let newScaleFactor = minMaxZoom(pinch.scale * lastZoomFactor)
    switch pinch.state {
    case .began, .changed:
      update(scale: newScaleFactor)
      self.labelZoom.text = String(format: "%.2fx", newScaleFactor)
      self.labelZoom.font = UIFont.preferredFont(forTextStyle: .title2)
    case .ended:
      lastZoomFactor = minMaxZoom(newScaleFactor)
      update(scale: lastZoomFactor)
      self.labelZoom.font = UIFont.preferredFont(forTextStyle: .body)
    default: break
    }
  }  // Pinch to Zoom End --------------------------------------------------------------------------------------------
}  // ViewController class End

extension ViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
    predict(sampleBuffer: sampleBuffer)
  }
}

// Programmatically save image
extension ViewController: AVCapturePhotoCaptureDelegate {
  func photoOutput(
    _ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?
  ) {
    if let error = error {
      print("error occurred : \(error.localizedDescription)")
    }
    if let dataImage = photo.fileDataRepresentation() {
      let dataProvider = CGDataProvider(data: dataImage as CFData)
      let cgImageRef: CGImage! = CGImage(
        jpegDataProviderSource: dataProvider!, decode: nil, shouldInterpolate: true,
        intent: .defaultIntent)
      var orientation = CGImagePropertyOrientation.right
      switch UIDevice.current.orientation {
      case .landscapeLeft:
        orientation = .up
      case .landscapeRight:
        orientation = .down
      default:
        break
      }
      var image = UIImage(cgImage: cgImageRef, scale: 0.5, orientation: .right)
      if let orientedCIImage = CIImage(image: image)?.oriented(orientation),
        let cgImage = CIContext().createCGImage(orientedCIImage, from: orientedCIImage.extent)
      {
        image = UIImage(cgImage: cgImage)
      }
      let imageView = UIImageView(image: image)
      imageView.contentMode = .scaleAspectFill
      imageView.frame = videoPreview.frame
      let imageLayer = imageView.layer
      videoPreview.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)

      let bounds = UIScreen.main.bounds
      UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
      self.View0.drawHierarchy(in: bounds, afterScreenUpdates: true)
      let img = UIGraphicsGetImageFromCurrentImageContext()
      UIGraphicsEndImageContext()
      imageLayer.removeFromSuperlayer()
      let activityViewController = UIActivityViewController(
        activityItems: [img!], applicationActivities: nil)
      activityViewController.popoverPresentationController?.sourceView = self.View0
      self.present(activityViewController, animated: true, completion: nil)
      //
      //            // Save to camera roll
      //            UIImageWriteToSavedPhotosAlbum(img!, nil, nil, nil);
    } else {
      print("AVCapturePhotoCaptureDelegate Error")
    }
  }
}
