//
//  VideoProcessingViewController.swift
//  YOLO
//
//  Created by Ethan Bootehsaz on 10/12/24.
//

import UIKit
import AVFoundation
import Vision
import CoreMedia
import CoreML
import AVKit

class VideoProcessingViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    // Model and Vision properties
    var mlModel = try! bestSmall(configuration: .init()).model
//    var mlModel = try! yolov8m(configuration: .init()).model
    var detector: VNCoreMLModel!
    var visionRequest: VNCoreMLRequest!

    // Color mapping for classes
    var colors: [String: UIColor] = [:]

    // Predictions from the model
    var currentPredictions: [VNRecognizedObjectObservation] = []

    // Variables for aspect ratio and orientation
    var videoOrientation: CGImagePropertyOrientation = .up
    var videoSize: CGSize = .zero

    // Bounding Box Views
    let maxBoundingBoxViews = 100
    var boundingBoxViews = [BoundingBoxView]()

    // Processed videos
    var processedVideos: [VideoItem] = []

    // IBOutlet for activity indicator
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!

    @IBOutlet weak var uploadButton: UIButton!
    // IBOutlet for collection view to display processed videos
    @IBOutlet weak var collectionView: UICollectionView!

    override func viewDidLoad() {
        super.viewDidLoad()
        setModel()
        setUpBoundingBoxViews()
        loadProcessedVideos()

        // Set up collection view
        collectionView.dataSource = self
        collectionView.delegate = self
        
        uploadButton.titleLabel?.adjustsFontSizeToFitWidth = true
        uploadButton.titleLabel?.numberOfLines = 1
        uploadButton.titleLabel?.minimumScaleFactor = 0.5
        uploadButton.titleLabel?.lineBreakMode = .byClipping
    }

    // MARK: - Model Setup

    func setModel() {

        /// VNCoreMLModel
        detector = try! VNCoreMLModel(for: mlModel)
        // detector.featureProvider = ThresholdProvider()
        detector.featureProvider = ThresholdProvider(iouThreshold: 0.45, confidenceThreshold: 0.3)

        /// VNCoreMLRequest
        let request = VNCoreMLRequest(
            model: detector,
            completionHandler: { [weak self] request, error in
                self?.processObservations(for: request, error: error)
            })
        request.imageCropAndScaleOption = .scaleFill  // Match the live video code
        visionRequest = request
    }

    // MARK: - Video Selection and Processing

    func selectVideoFromLibrary() {
        let imagePickerController = UIImagePickerController()
        imagePickerController.delegate = self
        imagePickerController.mediaTypes = ["public.movie"]
        imagePickerController.sourceType = .photoLibrary
        present(imagePickerController, animated: true, completion: nil)
    }

    @IBAction func uploadVideoButtonTapped(_ sender: UIButton) {
        selectVideoFromLibrary()
    }

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
        picker.dismiss(animated: true, completion: nil)
        if let videoURL = info[.mediaURL] as? URL {
            // Start processing the selected video
            processVideo(url: videoURL)
        } else {
            // Handle the error (e.g., show an alert)
            let alert = UIAlertController(title: "Error", message: "Failed to load video.", preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            present(alert, animated: true)
        }
    }

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }

    func processVideo(url: URL) {
        // Show the activity indicator
        DispatchQueue.main.async {
            self.activityIndicator.isHidden = false
            self.activityIndicator.startAnimating()
        }

        DispatchQueue.global(qos: .userInitiated).async {
            // Extract frames and process the video
            self.processVideoFrames(inputURL: url) { [weak self] outputURL in
                DispatchQueue.main.async {
                    self?.activityIndicator.stopAnimating()
                    self?.activityIndicator.isHidden = true
                    if let outputURL = outputURL {
                        // Optionally, play the processed video immediately
                        // self?.playVideo(url: outputURL)
                    } else {
                        // Handle error
                        let alert = UIAlertController(title: "Error", message: "Failed to process video.", preferredStyle: .alert)
                        alert.addAction(UIAlertAction(title: "OK", style: .default))
                        self?.present(alert, animated: true)
                    }
                }
            }
        }
    }

    func processVideoFrames(inputURL: URL, completion: @escaping (URL?) -> Void) {
        let asset = AVAsset(url: inputURL)
        let readerSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        guard let reader = try? AVAssetReader(asset: asset),
              let videoTrack = asset.tracks(withMediaType: .video).first else {
            completion(nil)
            return
        }

        let readerOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: readerSettings)
        reader.add(readerOutput)

        // Get video properties
        let naturalSize = videoTrack.naturalSize
        let transform = videoTrack.preferredTransform

        // Correct video orientation
        self.videoOrientation = self.videoOrientationFromTransform(transform)
        self.videoSize = naturalSize

        // Generate a unique output URL in the documents directory
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let outputURL = documentsDirectory.appendingPathComponent("processedVideo_\(UUID().uuidString).mp4")

        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else {
            completion(nil)
            return
        }

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: self.videoSize.width,
            AVVideoHeightKey: self.videoSize.height,
        ]

        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        writerInput.expectsMediaDataInRealTime = false
        writerInput.transform = transform  // Apply the original video transform

        let sourcePixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: self.videoSize.width,
            kCVPixelBufferHeightKey as String: self.videoSize.height,
        ]

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: sourcePixelBufferAttributes
        )

        writer.add(writerInput)

        writer.startWriting()
        reader.startReading()
        writer.startSession(atSourceTime: CMTime.zero)

        let processingQueue = DispatchQueue(label: "videoProcessingQueue")

        // Prepare to process frames
        writerInput.requestMediaDataWhenReady(on: processingQueue) {
            while writerInput.isReadyForMoreMediaData {
                if let sampleBuffer = readerOutput.copyNextSampleBuffer() {
                    autoreleasepool {
                        if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                            let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
                            // Process the frame
                            let processedBuffer = self.processFrame(pixelBuffer: pixelBuffer)

                            // Append to writer
                            adaptor.append(processedBuffer, withPresentationTime: presentationTime)
                        }
                    }
                } else {
                    writerInput.markAsFinished()
                    writer.finishWriting {
                        if writer.status == .completed {
                            // Generate thumbnail
                            let thumbnail = self.generateThumbnail(url: outputURL)

                            // Create VideoItem
                            let videoItem = VideoItem(url: outputURL, thumbnail: thumbnail, date: Date())

                            // Save the VideoItem
                            self.saveProcessedVideo(videoItem)

                            completion(outputURL)
                        } else {
                            completion(nil)
                        }
                    }
                    break
                }
            }
        }
    }

    func processFrame(pixelBuffer: CVPixelBuffer) -> CVPixelBuffer {
        // Run the model on the frame and overlay predictions

        // Run model prediction
        self.predict(pixelBuffer: pixelBuffer)

        // Draw bounding boxes on the image
        let processedBuffer = self.drawPredictions(predictions: currentPredictions, onPixelBuffer: pixelBuffer)

        return processedBuffer
    }

    func predict(pixelBuffer: CVPixelBuffer) {
        // Set up the VNImageRequestHandler and perform the vision request
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: self.videoOrientation, options: [:])

        do {
            try handler.perform([visionRequest])
            // The results will be handled in the completion handler
        } catch {
            print("Error performing prediction: \(error)")
        }
    }

    func processObservations(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            if let results = request.results as? [VNRecognizedObjectObservation] {
                // Filter predictions to only include those labeled 'person'
                self.currentPredictions = results.filter { prediction in
                    if let bestLabel = prediction.labels.first {
                        return bestLabel.identifier == "person"
                    }
                    return false
                }
            } else {
                self.currentPredictions = []
            }
        }
    }

    func drawPredictions(predictions: [VNRecognizedObjectObservation], onPixelBuffer pixelBuffer: CVPixelBuffer) -> CVPixelBuffer {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let uiImage = UIImage(ciImage: ciImage)

        UIGraphicsBeginImageContextWithOptions(uiImage.size, false, uiImage.scale)
        let context = UIGraphicsGetCurrentContext()!

        // Draw the original image
        uiImage.draw(at: CGPoint.zero)

        let width = uiImage.size.width
        let height = uiImage.size.height

        // Draw the bounding boxes
        context.saveGState()
        for prediction in predictions {
            var rect = prediction.boundingBox  // Normalized coordinates (origin at bottom-left)

            // Adjust bounding box based on orientation
            rect = self.adjustBoundingBox(rect, forOrientation: self.videoOrientation)

            // Convert normalized coordinates to pixel coordinates
            rect = CGRect(
                x: rect.origin.x * width,
                y: rect.origin.y * height,
                width: rect.width * width,
                height: rect.height * height
            )

            let color = colors["person"] ?? UIColor.red  // Use color for 'person' label
            context.setStrokeColor(color.cgColor)
            context.setLineWidth(2.0)
            context.stroke(rect)

            // Draw label
            let bestClass = prediction.labels.first?.identifier ?? ""
            let confidence = prediction.labels.first?.confidence ?? 0.0
            let label = String(format: "%@ %.1f", bestClass, confidence)

            let textAttributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 16),
                .foregroundColor: UIColor.white,
                .backgroundColor: color
            ]
            let textSize = label.size(withAttributes: textAttributes)
            let textOrigin = CGPoint(x: rect.origin.x, y: rect.origin.y - textSize.height)
            let textRect = CGRect(origin: textOrigin, size: textSize)
            label.draw(in: textRect, withAttributes: textAttributes)
        }
        context.restoreGState()

        // Get the new image
        let processedUIImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        // Convert back to pixel buffer
        let processedCIImage = CIImage(image: processedUIImage!)!
        let contextCI = CIContext(options: nil)

        var outputPixelBuffer: CVPixelBuffer?

        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary

        CVPixelBufferCreate(kCFAllocatorDefault, Int(width), Int(height), kCVPixelFormatType_32BGRA, attrs, &outputPixelBuffer)

        contextCI.render(processedCIImage, to: outputPixelBuffer!)

        return outputPixelBuffer!
    }

    func adjustBoundingBox(_ rect: CGRect, forOrientation orientation: CGImagePropertyOrientation) -> CGRect {
        var adjustedRect = rect

        switch orientation {
        case .up:
            // No change
            adjustedRect = CGRect(
                x: rect.origin.x,
                y: 1.0 - rect.origin.y - rect.height,
                width: rect.width,
                height: rect.height)
        case .down:
            adjustedRect = CGRect(
                x: 1.0 - rect.origin.x - rect.width,
                y: rect.origin.y,
                width: rect.width,
                height: rect.height)
        case .left:
            adjustedRect = CGRect(
                x: rect.origin.y,
                y: rect.origin.x,
                width: rect.height,
                height: rect.width)
        case .right:
            adjustedRect = CGRect(
                x: 1.0 - rect.origin.y - rect.height,
                y: 1.0 - rect.origin.x - rect.width,
                width: rect.height,
                height: rect.width)
        default:
            // Handle other orientations if necessary
            break
        }

        return adjustedRect
    }

    func playVideo(url: URL) {
        let player = AVPlayer(url: url)
        let playerViewController = AVPlayerViewController()
        playerViewController.player = player
        present(playerViewController, animated: true) {
            player.play()
        }
    }

    // MARK: - Helper Methods

    func setUpBoundingBoxViews() {
        // Ensure all bounding box views are initialized up to the maximum allowed.
        while boundingBoxViews.count < maxBoundingBoxViews {
            boundingBoxViews.append(BoundingBoxView())
        }

        // Assign a color specifically for 'person' label
        colors["person"] = UIColor(
            red: CGFloat.random(in: 0...1),
            green: CGFloat.random(in: 0...1),
            blue: CGFloat.random(in: 0...1),
            alpha: 0.9)
    }

    func setUpBoundingBoxColors() {
        // This function is redundant since colors are set up in setUpBoundingBoxViews()
    }

    func videoOrientationFromTransform(_ transform: CGAffineTransform) -> CGImagePropertyOrientation {
        if transform.a == 0 && transform.b == 1.0 && transform.c == -1.0 && transform.d == 0 {
            return .right  // Portrait
        } else if transform.a == 0 && transform.b == -1.0 && transform.c == 1.0 && transform.d == 0 {
            return .left   // PortraitUpsideDown
        } else if transform.a == 1.0 && transform.b == 0 && transform.c == 0 && transform.d == 1.0 {
            return .up     // LandscapeRight
        } else if transform.a == -1.0 && transform.b == 0 && transform.c == 0 && transform.d == -1.0 {
            return .down   // LandscapeLeft
        } else {
            return .up
        }
    }

    // MARK: - Processed Videos Management

    func generateThumbnail(url: URL) -> UIImage {
        let asset = AVAsset(url: url)
        let imageGenerator = AVAssetImageGenerator(asset: asset)
        imageGenerator.appliesPreferredTrackTransform = true

        let time = CMTime(seconds: 1.0, preferredTimescale: 600)
        var actualTime = CMTime.zero
        let imageRef: CGImage

        do {
            imageRef = try imageGenerator.copyCGImage(at: time, actualTime: &actualTime)
        } catch {
            print("Error generating thumbnail: \(error)")
            return UIImage()
        }

        return UIImage(cgImage: imageRef)
    }

    func saveProcessedVideo(_ videoItem: VideoItem) {
        DispatchQueue.main.async {
            // Save to the array
            self.processedVideos.append(videoItem)
            // Save to persistent storage
            self.saveVideosToUserDefaults()
            // Reload collection view
            self.collectionView.reloadData()
        }
    }

    func saveVideosToUserDefaults() {
        let encoder = JSONEncoder()
        if let encoded = try? encoder.encode(processedVideos.map { $0.url.absoluteString }) {
            UserDefaults.standard.set(encoded, forKey: "processedVideos")
        }
    }

    func loadProcessedVideos() {
        if let savedData = UserDefaults.standard.data(forKey: "processedVideos") {
            let decoder = JSONDecoder()
            if let savedURLs = try? decoder.decode([String].self, from: savedData) {
                self.processedVideos = savedURLs.compactMap { urlString in
                    if let url = URL(string: urlString), FileManager.default.fileExists(atPath: url.path) {
                        let thumbnail = self.generateThumbnail(url: url)
                        return VideoItem(url: url, thumbnail: thumbnail, date: Date())
                    }
                    return nil
                }
            }
        }
    }
}

// MARK: - UICollectionView DataSource and Delegate

extension VideoProcessingViewController: UICollectionViewDataSource, UICollectionViewDelegateFlowLayout {

    // MARK: - UICollectionViewDataSource Methods

    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return processedVideos.count
    }

    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {

        // Dequeue a cell
        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "VideoCell", for: indexPath) as! VideoCell

        // Configure the cell
        let videoItem = processedVideos[indexPath.item]
        cell.imageView.image = videoItem.thumbnail
        cell.dateLabel.text = DateFormatter.localizedString(from: videoItem.date, dateStyle: .short, timeStyle: .short)

        return cell
    }

    // MARK: - UICollectionViewDelegateFlowLayout Methods

    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        let videoItem = processedVideos[indexPath.item]
        let player = AVPlayer(url: videoItem.url)
        let playerViewController = AVPlayerViewController()
        playerViewController.player = player
        present(playerViewController, animated: true) {
            player.play()
        }
    }

    // Optional: Adjust cell size
    func collectionView(_ collectionView: UICollectionView,
                        layout collectionViewLayout: UICollectionViewLayout,
                        sizeForItemAt indexPath: IndexPath) -> CGSize {
        // Adjust the size as needed
        let width = (collectionView.frame.width - 20) / 2  // Two columns with spacing
        return CGSize(width: width, height: width * 1.2)    // Adjust the height ratio as needed
    }
}


struct VideoItem {
    let url: URL
    let thumbnail: UIImage
    let date: Date
}
