package com.example.basmati

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff

import android.icu.number.Scale
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.basmati.ml.Best5Float32
import org.opencv.android.Utils
import org.opencv.android.Utils.bitmapToMat
import org.opencv.android.Utils.matToBitmap
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MLModelHandler(context: Context) {
    private var interpreter: Interpreter? = null

    init {
        loadModel(context)
    }

    private fun loadModel(context: Context) {
        val assetFilePath = "best5_float32.tflite" // Replace with your model name
        val fileDescriptor = context.assets.openFd(assetFilePath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val mappedByteBuffer: MappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.length)

        interpreter = Interpreter(mappedByteBuffer)
        inputStream.close() // Ensure you close the stream
    }
    fun preprocessImage(input: Bitmap, inputSize: Int): Array<Array<Array<FloatArray>>> {
        // Resize the image to the required size
        val resizedImage = Bitmap.createScaledBitmap(input, inputSize, inputSize, true)

        // Create a 4D array [1, inputSize, inputSize, 3] for 3 channels (RGB)
        val inputBuffer = Array(1) { Array(inputSize) { Array(inputSize) { FloatArray(3) } } }

        // Convert the Bitmap to a FloatArray (RGB channels)
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val pixel = resizedImage.getPixel(x, y)

                // Extract RGB components (0-255) and normalize them to [0, 1]
                val r = Color.red(pixel) / 255f
                val g = Color.green(pixel) / 255f
                val b = Color.blue(pixel) / 255f

                // Populate the input buffer with RGB values
                inputBuffer[0][y][x][0] = r // Red channel
                inputBuffer[0][y][x][1] = g // Green channel
                inputBuffer[0][y][x][2] = b // Blue channel
            }
        }

        return inputBuffer
    }




    fun runInference(input: Bitmap): Array<Array<FloatArray>> {
        val inputSize = 640 // Replace with your model's input size
        val inputBuffer = preprocessImage(input, inputSize)

        val outputBuffer = Array(1) { Array(5) { FloatArray(8400) } } // Adjust based on your model's output size

        // Run inference
        interpreter?.run(inputBuffer, outputBuffer)

        return outputBuffer
    }
}




class MainActivity : AppCompatActivity() {

    private lateinit var mlModelHandler: MLModelHandler
    lateinit var imageView: ImageView
    lateinit var original: Bitmap
    lateinit var button : Button
    lateinit var model: Best5Float32

    var imageProcessor = ImageProcessor.Builder().add(ResizeOp(640,640,ResizeOp.ResizeMethod.BILINEAR)).build()

    data class Box(val x: Float, val y: Float, val width: Float, val height: Float)

    data class Detection(val box: Box, val confidence: Float)


    override fun onCreate(savedInstanceState: Bundle?) {
        System.loadLibrary("opencv_java4") // or "opencv_java" based on your version

        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        mlModelHandler = MLModelHandler(this)


        val intent = Intent()
        intent.setType("image/*")
        intent.setAction(Intent.ACTION_GET_CONTENT)

        model = Best5Float32.newInstance(this)

//        imageView = findViewById(R.id.imaegView)
        button = findViewById(R.id.btn)

        button.setOnClickListener{
            startActivityForResult(intent, requestCode = 101)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode==101){
            var uri = data?.data
            original = MediaStore.Images.Media.getBitmap(this.contentResolver,uri)

            val originalImageView: ImageView = findViewById(R.id.original_image_view)
            originalImageView.setImageBitmap(original)

            get_prediction();
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }






    fun adjustImageOrientation(bitmap: Bitmap): Bitmap {
        // Check if the width is greater than the height
        return if (bitmap.width > bitmap.height) {
            // Rotate the image 270 degrees
            val matrix = Matrix().apply {
                postRotate(-270f)
            }
            // Create a new bitmap with the applied rotation
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            // Return the original image if no rotation is needed
            bitmap
        }
    }

    fun get_prediction(){

        var adjustedBitmap = adjustImageOrientation(original)
        if (adjustedBitmap == null) {
            Log.e("Error", "Adjusted bitmap is null")
            return
        }
        var matImage = Mat()
        Utils.bitmapToMat(adjustedBitmap,matImage)
        if (matImage.empty()) {
            Log.e("Error", "Mat image is empty after conversion from bitmap")
            return
        }

        Imgproc.cvtColor(matImage,matImage,Imgproc.COLOR_RGB2BGR)
//        val newsize = Size(800.0,600.0)
//        Imgproc.resize(matImage,matImage,newsize)

        val grayImage = Mat()
        Imgproc.cvtColor(matImage,grayImage,Imgproc.COLOR_BGR2GRAY)

        val blurredImage = Mat()
        Imgproc.GaussianBlur(grayImage,blurredImage,Size(7.0,7.0),0.0)

        val lower = Scalar(0.0)
        val upper = Scalar(160.0)
        val mask =Mat()
        Core.inRange(blurredImage,lower,upper,mask)

        val thresh = Mat()
        Imgproc.threshold(mask,thresh,1.0,255.0,Imgproc.THRESH_BINARY)

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(thresh,contours,hierarchy,Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE)

        val areas = contours.map{Imgproc.contourArea(it)}
        val maxIndex = areas.indices.maxByOrNull { areas[it] } ?: -1

        if (maxIndex != -1){
            val cnt = contours[maxIndex]
            val rect = Imgproc.minAreaRect(MatOfPoint2f(*cnt.toArray()))
            val rectPoints = arrayOfNulls<Point>(4)
            rect.points(rectPoints)

            val rectPointsInt = rectPoints.mapNotNull { point ->
                point?.let { Point(it.x.toInt().toDouble(), it.y.toInt().toDouble()) }
            }

            val contoursSorted = contours.sortedByDescending { Imgproc.contourArea(it) }
            var epsilon = 0.05 * Imgproc.arcLength(MatOfPoint2f(*contoursSorted[2].toArray()), true)
            var approx = MatOfPoint2f()
            Imgproc.approxPolyDP(MatOfPoint2f(*contoursSorted[2].toArray()), approx, epsilon, true)

            if (approx.toArray().size != 4) {
                epsilon = 0.05 * Imgproc.arcLength(MatOfPoint2f(*contoursSorted[1].toArray()), true)
                Imgproc.approxPolyDP(MatOfPoint2f(*contoursSorted[1].toArray()), approx, epsilon, true)
            }

            if (approx.toArray().size != 4) {
                epsilon = 0.05 * Imgproc.arcLength(MatOfPoint2f(*contoursSorted[0].toArray()), true)
                Imgproc.approxPolyDP(MatOfPoint2f(*contoursSorted[0].toArray()), approx, epsilon, true)
            }

            // 13. Sort coordinates of the approximated polygon
            val sortedCoords = approx.toList().sortedWith(compareBy { it.x + it.y })


//            val sorcor = mutableListOf<Point>(sortedCoords[0],sortedCoords[1],sortedCoords[2],sortedCoords[3])

            var bottomLeft = sortedCoords[0]
//            sorcor.removeAt(0)
            var topRight = sortedCoords[3]
//            sorcor.removeAt(2)

            val remainingCoords = mutableListOf<Point>(sortedCoords[1], sortedCoords[2])

            val selectedCoordinate: Point
            if (remainingCoords[0].y > remainingCoords[1].y) {
                selectedCoordinate = remainingCoords[0]
                remainingCoords.removeAt(0)
            } else {
                selectedCoordinate = remainingCoords[1]
                remainingCoords.removeAt(1)
            }

            val topLeft = selectedCoordinate
            val bottomRight = remainingCoords[0]

            // 14. Set up points A, B, C, and D
            val pt_A = bottomLeft
            val pt_B = topLeft
            val pt_C = topRight
            val pt_D = bottomRight

            val width_AD = Math.sqrt(Math.pow((pt_A.x - pt_D.x), 2.0) + Math.pow((pt_A.y - pt_D.y), 2.0))
            val width_BC = Math.sqrt(Math.pow((pt_B.x - pt_C.x), 2.0) + Math.pow((pt_B.y - pt_C.y), 2.0))
            val maxWidth = Math.max(width_AD.toInt(), width_BC.toInt())

            val height_AB = Math.sqrt(Math.pow((pt_A.x - pt_B.x), 2.0) + Math.pow((pt_A.y - pt_B.y), 2.0))
            val height_CD = Math.sqrt(Math.pow((pt_C.x - pt_D.x), 2.0) + Math.pow((pt_C.y - pt_D.y), 2.0))
            val maxHeight = Math.max(height_AB.toInt(), height_CD.toInt())

            // 16. Define input and output points for perspective transform
            val inputPts = MatOfPoint2f(pt_A, pt_B, pt_C, pt_D)
            val outputPts = MatOfPoint2f(
                Point(0.0, 0.0),
                Point(0.0, (maxHeight - 1).toDouble()),
                Point((maxWidth - 1).toDouble(), (maxHeight - 1).toDouble()),
                Point((maxWidth - 1).toDouble(), 0.0)
            )

            val M = Imgproc.getPerspectiveTransform(inputPts, outputPts)
            val warpedImage = Mat()
            Imgproc.warpPerspective(matImage, warpedImage, M, Size(maxWidth.toDouble(), maxHeight.toDouble()), Imgproc.INTER_LINEAR)

            // 18. Resize the final image to a specific size (e.g., 5412x6142)
            val finalSize = Size(5412.0, 6142.0)
            Imgproc.resize(warpedImage, warpedImage, finalSize, 0.0, 0.0, Imgproc.INTER_CUBIC)


            var processedBitmap = Bitmap.createBitmap(warpedImage.cols(), warpedImage.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(warpedImage, processedBitmap)


//            var test = Mat()
//            Utils.bitmapToMat(processedBitmap,test)
//            val modelsize = Size(640.0,640.0)
//            Imgproc.resize(test,test,modelsize)
//            var modelbitmap = Bitmap.createBitmap(test.cols(),test.rows(),Bitmap.Config.ARGB_8888)
//            Utils.matToBitmap(test,modelbitmap)



            try {
//                val model = Best5Float32.newInstance(this)
//
//                val inputImage = TensorImage.fromBitmap(modelbitmap)
//
////                inputImage.load(processedBitmap)
//
//
//                val outputs = model.process(inputImage)// Assuming model is an instance of the Interpreter or the TFLite model
//                val outputtensorbuffer = outputs.outputAsTensorBuffer
//                val outputarray = outputtensorbuffer.floatArray
//                val numdetection = outputarray.size/5
//
////                for (i in outputarray){
////                    println(i)
////                }
//
//
//

                val output = mlModelHandler.runInference(processedBitmap)
                val output2D = Array(5) { FloatArray(8400) }
                for (i in 0 until 5) {
                    for (j in 0 until 8400) {
                        output2D[i][j] = output[0][i][j]
                    }
                }
                println("Size of output2d: ${output2D[0].size}")

                val (boxes, confidences) = decodeModelOutput(output2D)

                println("Size of boxes: ${boxes.size}")
                println("Size of confidences: ${confidences.size}")


                val boxObjects = boxes.map { box ->
                    Box(box[0], box[1], box[2], box[3])
                }
                val selectedIndices = nonMaximumSuppression(boxObjects, confidences, scoreThreshold = 0.5f, nmsThreshold = 0.5f)

                // Filter the boxes and confidences based on NMS results
                val filteredBoxes = selectedIndices.map { boxObjects[it] }
                val filteredConfidences = selectedIndices.map { confidences[it] }
                println("Size of filteredboxes: ${filteredBoxes.size}")

                val imageWidth = 5412f
                val imageHeight = 6142f

                // Initialize the list to store the final coordinates
                val filteredCoords = mutableListOf<List<Int>>()

                // Process the filtered boxes and calculate the final coordinates
                for (box in filteredBoxes) {
                    val x = box.x
                    val y = box.y
                    val w = box.width
                    val h = box.height

                    // Calculate the start and end coordinates
                    val startX = (x - (w / 2)) * imageWidth
                    val startY = (y - (h / 2)) * imageHeight
                    val endX = (x + (w / 2)) * imageWidth
                    val endY = (y + (h / 2)) * imageHeight

                    // Convert the coordinates to integers and add them to the list
                    val startXInt = Math.max(0,startX.toInt())
                    val startYInt = Math.max(0,startY.toInt())
                    val endXInt = Math.min(5412,endX.toInt())
                    val endYInt = Math.min(6142,endY.toInt())

                    // Add the calculated coordinates to the filteredCoords list
                    filteredCoords.add(listOf(startXInt, startYInt, endXInt, endYInt))
                }
//                for (coords in filteredCoords) {
//                    Log.d("Filtered Coords", "Coordinates: (${coords[0]}, ${coords[1]}, ${coords[2]}, ${coords[3]})")
//                }

                val rectList = mutableListOf<Rect>()
                for (coords in filteredCoords) {
                    val startX = coords[0]
                    val startY = coords[1]
                    val endX = coords[2]
                    val endY = coords[3]

                    // Calculate width and height
                    val width = endX - startX
                    val height = endY - startY

                    // Create a Rect using top-left corner (startX, startY) and dimensions (width, height)
                    val rect = Rect(startX, startY, width, height)
                    rectList.add(rect)
                }

                val inputImageMat = Mat()
                Utils.bitmapToMat(processedBitmap, inputImageMat)

//                val processedImageMat = processPrediction(inputImageMat, rectList)



                if (inputImageMat.channels() == 4) {
                    Imgproc.cvtColor(inputImageMat, inputImageMat, Imgproc.COLOR_BGRA2BGR);
                }

//                Imgproc.putText(inputImageMat, "Test Text", Point(1000.0, 1000.0), Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(0.0, 255.0, 0.0), 10)
                val riceType = mutableListOf<String>()
                val chalkiness = mutableListOf<String>()
                val pos = mutableListOf<Int>()
                val ar = mutableListOf<Float>()
                val lengths = mutableListOf<Float>()
                val breadths = mutableListOf<Double>()
                val index = mutableListOf<Int>()
                var tbas = 0
                var tnbas = 0

                // Scaling multipliers
                val mult1 = 60.0 / 6142
                val mult2 = 60.0 / 5412

                for ((i,rectangle) in rectList.withIndex()){
                    val startX = rectangle.x
                    val startY = rectangle.y
                    val endX = startX + rectangle.width
                    val endY = startY + rectangle.height

                    val roi = inputImageMat.submat(rectangle)
                    val imgGray = Mat()
                    val imgBlur = Mat()
                    Imgproc.cvtColor(roi, imgGray, Imgproc.COLOR_BGR2GRAY)
                    Imgproc.GaussianBlur(imgGray, imgBlur, Size(7.0, 7.0), 0.0,0.0)

                    // Thresholding (ensure the thresholded image is 8-bit single-channel)
                    val thresh = Mat()
                    Imgproc.threshold(imgBlur, thresh, 155.0, 255.0, Imgproc.THRESH_BINARY)
                    val whitePixels = Core.countNonZero(thresh)
                    val totalPixels = thresh.size()
                    val whiteProportion = (whitePixels / (thresh.cols()+thresh.rows())).toFloat()

//                    println("totalpixels : ${thresh.cols()*thresh.rows()}")
//                    println("white : ${whiteProportion}")
                    if (whiteProportion >= 0.17f) {

//                        println("white : ${whiteProportion}")
                        // Find contours (ensure contours are properly extracted from binary image)
                        val contours = mutableListOf<MatOfPoint>()
                        val hierarchy = Mat()
                        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)

                        if (contours.isNotEmpty()) {
                            // Find the largest contour
                            val areas = contours.map { contour -> Imgproc.contourArea(contour) }

// Find the index of the maximum area
                            val maxIndex = areas.indices.maxByOrNull { areas[it] } ?: -1

// Get the contour with the largest area
                            val cnt = if (maxIndex != -1) contours[maxIndex] else null
//                    val largestContour = contours.maxByOrNull { contour -> Imgproc.contourArea(contour) } ?: continue

                            // Calculate moments and centroid
                            val moments = Imgproc.moments(cnt)
                            val centroidX = (moments.m10 / moments.m00).toInt()
                            val centroidY = (moments.m01 / moments.m00).toInt()

                            // Adjust Y bounds using vertical line scan
//                    val height = thresh.rows()

                            var y1_i = centroidY
                            var y2_i = centroidY

                            // Scan downwards from centroid to find lower boundary
                            var k = 0
                            while (k + centroidY < thresh.rows() - 1) {
                                if (thresh[centroidY + k, centroidX][0] != thresh[centroidY + k + 1, centroidX][0]) {
                                    y1_i = centroidY + k
                                    break
                                }
                                k++
                            }

                            // Scan upwards from centroid to find upper boundary
                            var j = 0
                            while (centroidY - j >= 0) {
                                if (thresh[centroidY - j, centroidX][0] != thresh[centroidY - j - 1, centroidX][0]) {
                                    y2_i = centroidY - j
                                    break
                                }
                                j++
                            }

                            // Find minimum area rectangle
                            val rect = cnt?.let {
                                Imgproc.minAreaRect(MatOfPoint2f(*it.toArray()))
                            } ?: throw IllegalArgumentException("Contours cannot be null")

// Continue processin
                            val size = rect.size
                            var h1 = size.height.toFloat()
                            var w1 = size.width.toFloat()


                            var l1: Double
                            var b1: Double

                            if (h1 >= w1) {
                                l1 = h1 * mult2
                                b1 = w1 * mult1
                            } else {
                                l1 = w1 * mult2
                                b1 = h1 * mult1
                                // Swap h1 and w1
                                val temp = h1
                                h1 = w1
                                w1 = temp
                            }


                            val boundingRect = Imgproc.boundingRect(cnt)
                            val x1 = boundingRect.x
                            val y1 = boundingRect.y


                            b1 = (y1_i - y2_i) * mult1
                            breadths.add(b1)

                            val rectPoints = arrayOf(Point(), Point(), Point(), Point())
                            rect.points(rectPoints)

// Step 3: Convert points to integer format (like np.int0 in Python)
                            val adjustedBox = rectPoints.map { Point(it.x.toInt().toDouble(), it.y.toInt().toDouble()) }.toTypedArray()

// Step 4: Adjust x and y coordinates of the bounding box
                            adjustedBox.forEach {
                                it.x += startX.toDouble()  // Adjust x-coordinates
                                it.y += startY.toDouble()  // Adjust y-coordinates
                            }

// Step 5: Draw the adjusted bounding box on the image
                            val matOfPoint = MatOfPoint(*adjustedBox)
                            Imgproc.drawContours(inputImageMat, listOf(matOfPoint), 0, Scalar(0.0, 255.0, 0.0), 10)

                            val aspectRatio = h1.toFloat() / w1
                            ar.add(aspectRatio)
                            index.add(i)

                            val thresh1 = Mat()
                            val ret1 = Imgproc.threshold(imgBlur,thresh1, 207.0, 255.0, Imgproc.THRESH_BINARY) // imgBlur is the input Mat

                            val white1 = Core.countNonZero(thresh1)
                            val total1 = thresh1.total().toInt() // total() returns the total number of pixels
                            val whiteProportion1 = white1.toDouble() / (thresh1.cols()+thresh1.rows())

                            if (l1 < 6.61) {
                                riceType.add("Non-Basmati_$i")
                                Imgproc.putText(inputImageMat, i.toString(),
                                    Point((startX+x1).toDouble(),(startY+y1+100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)
                                Imgproc.putText(inputImageMat, "NB",
                                    Point((startX + x1 + 220).toDouble(), (startY + y1 + 100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)
                                tnbas += 1


                                if (whiteProportion1 >= 0.02) {
                                    chalkiness.add("chalky_$i")
                                    Imgproc.putText(inputImageMat, "C", Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)

                                } else {
                                    chalkiness.add("non-chalky_$i")
                                    Imgproc.putText(inputImageMat, "NC",
                                        Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)
                                }
                            }else if (b1 > 2.01) {
                                riceType.add("Non-Basmati_$i")
                                Imgproc.putText(inputImageMat, i.toString(),
                                    Point((startX+x1).toDouble(),(startY+y1+100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)
                                Imgproc.putText(inputImageMat, "NB",
                                    Point((startX + x1 + 220).toDouble(), (startY + y1 + 100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)
                                tnbas += 1


                                if (whiteProportion1 >= 0.02) {
                                    chalkiness.add("chalky_$i")
                                    Imgproc.putText(inputImageMat, "C", Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)

                                } else {
                                    chalkiness.add("non-chalky_$i")
                                    Imgproc.putText(inputImageMat, "NC",
                                        Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)
                                }
                            } else if (aspectRatio < 3.5) {
                                riceType.add("Non-Basmati_$i")
                                Imgproc.putText(inputImageMat, i.toString(),
                                    Point((startX+x1).toDouble(),(startY+y1+100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)
                                Imgproc.putText(inputImageMat, "NB",
                                    Point((startX + x1 + 220).toDouble(), (startY + y1 + 100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)
                                tnbas += 1


                                if (whiteProportion1 >= 0.02) {
                                    chalkiness.add("chalky_$i")
                                    Imgproc.putText(inputImageMat, "C", Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)

                                } else {
                                    chalkiness.add("non-chalky_$i")
                                    Imgproc.putText(inputImageMat, "NC",
                                        Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)
                                }
                            } else {
                                riceType.add("Basmati_$i")
                                Imgproc.putText(inputImageMat, i.toString(),
                                    Point((startX+x1).toDouble(),(startY+y1+100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(0.0, 0.0, 255.0), 16)
                                Imgproc.putText(inputImageMat, "B",
                                    Point((startX + x1 + 220).toDouble(), (startY + y1 + 100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(0.0, 0.0, 255.0), 16)
                                tbas += 1


                                if (whiteProportion1 >= 0.02) {
                                    chalkiness.add("chalky_$i")
                                    Imgproc.putText(inputImageMat, "C", Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(0.0, 0.0, 255.0), 16)

                                } else {
                                    chalkiness.add("non-chalky_$i")
                                    Imgproc.putText(inputImageMat, "NC",
                                        Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(0.0, 0.0, 255.0), 16)
                                }
                            }

                        }
                    }


                }



                val resultbitmap = Bitmap.createBitmap(inputImageMat.cols(),inputImageMat.rows(),Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(inputImageMat,resultbitmap)

                imageView = findViewById(R.id.processed_image_view)
                imageView.setImageBitmap(resultbitmap)
                saveImageToGallery(resultbitmap, "final Image", "This is a processed image")


                // Set the desired text
                val adulteration = tnbas * 100.0 / (tbas + tnbas)
                val roundedAdulteration = String.format("%.2f", adulteration)

                val processedImageTextView: TextView = findViewById(R.id.processed_image_text)
                val desiredText = "Total Rice Grains detected : ${tbas+tnbas} \n No. of Basmati grains : ${tbas} \n No. of Non-Basmati grains : ${tnbas} \n Adulteration : ${roundedAdulteration}%."
                processedImageTextView.text = desiredText


            } catch (e: IOException) {
                e.printStackTrace()
            }

        }

    }

    fun processPrediction(inputImage: Mat, filteredCoords: List<Rect>): Mat {
        // Copy the input image
        var drawonthis = inputImage.clone()

        // Convert the image to a compatible type for contour processing (CV_8UC3 for color images)
        if (drawonthis.channels() == 4) {
            Imgproc.cvtColor(drawonthis, drawonthis, Imgproc.COLOR_RGBA2BGR)
        }

// Log after conversion to check
        Log.d("Mat Type After Conversion", "Mat Type: ${drawonthis.type()}")
        Log.d("Mat Channels After Conversion", "Mat Channels: ${drawonthis.channels()}")

//        var canvasBitmap = Bitmap.createBitmap(500, 500, Bitmap.Config.ARGB_8888)

        val testMat = Mat(6142, 5412, CvType.CV_8UC3, Scalar(255.0, 255.0, 255.0)) // Create a white canvas
        Imgproc.putText(testMat, "Test Text", Point(50.0, 50.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0.0, 0.0, 255.0), 10) // Draw red text

// Convert to Bitmap



        // Initialize required variables
        val riceType = mutableListOf<String>()
        val chalkiness = mutableListOf<String>()
        val pos = mutableListOf<Int>()
        val ar = mutableListOf<Float>()
        val lengths = mutableListOf<Float>()
        val breadths = mutableListOf<Double>()
        val index = mutableListOf<Int>()
        var tbas = 0
        var tnbas = 0

        // Scaling multipliers
        val mult1 = 60.0 / 6142
        val mult2 = 60.0 / 5412


//        var finalbitmap = Bitmap.createBitmap(drawonthis.cols(), drawonthis.rows(), Bitmap.Config.ARGB_8888)
//        Utils.matToBitmap(drawonthis, finalbitmap)
//
//        // Create a Canvas from the Bitmap
//        var canvas = Canvas(finalbitmap)
//        var paint = Paint()
//        paint.textSize = 50f


        for ((i, rect) in filteredCoords.withIndex()) {
            // Get coordinates and ROI from the image
            val startX = rect.x
            val startY = rect.y
            val endX = startX + rect.width
            val endY = startY + rect.height

            // Extract region of interest (ROI) from the image
            val roi = drawonthis.submat(rect)

            // Convert to grayscale and apply Gaussian blur
            val imgGray = Mat()
            val imgBlur = Mat()
            Imgproc.cvtColor(roi, imgGray, Imgproc.COLOR_BGR2GRAY)
            Imgproc.GaussianBlur(imgGray, imgBlur, Size(7.0, 7.0), 0.0)

            // Thresholding (ensure the thresholded image is 8-bit single-channel)
            val thresh = Mat()
            Imgproc.threshold(imgBlur, thresh, 155.0, 255.0, Imgproc.THRESH_BINARY)

            // Convert thresholded image to the correct type (CV_8U)
//            if (thresh.type() != CvType.CV_8U) {
//                thresh.convertTo(thresh, CvType.CV_8U)
//            }

            // Calculate white pixel proportion
            val whitePixels = Core.countNonZero(thresh)
            val totalPixels = thresh.total().toInt()
            val whiteProportion = (whitePixels / totalPixels).toFloat()

            if (whiteProportion >= 0.17f) {
                // Find contours (ensure contours are properly extracted from binary image)
                val contours = mutableListOf<MatOfPoint>()
                val hierarchy = Mat()
                Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)

                if (contours.isNotEmpty()) {
                    // Find the largest contour
                    val areas = contours.map { contour -> Imgproc.contourArea(contour) }

// Find the index of the maximum area
                    val maxIndex = areas.indices.maxByOrNull { areas[it] } ?: -1

// Get the contour with the largest area
                    val cnt = if (maxIndex != -1) contours[maxIndex] else null
//                    val largestContour = contours.maxByOrNull { contour -> Imgproc.contourArea(contour) } ?: continue

                    // Calculate moments and centroid
                    val moments = Imgproc.moments(cnt)
                    val centroidX = (moments.m10 / moments.m00).toInt()
                    val centroidY = (moments.m01 / moments.m00).toInt()

                    // Adjust Y bounds using vertical line scan
//                    val height = thresh.rows()

                    var y1_i = centroidY
                    var y2_i = centroidY

                    // Scan downwards from centroid to find lower boundary
                    var k = 0
                    while (k + centroidY < thresh.rows() - 1) {
                        if (thresh[centroidY + k, centroidX][0] != thresh[centroidY + k + 1, centroidX][0]) {
                            y1_i = centroidY + k
                            break
                        }
                        k++
                    }

                    // Scan upwards from centroid to find upper boundary
                    var j = 0
                    while (centroidY - j >= 0) {
                        if (thresh[centroidY - j, centroidX][0] != thresh[centroidY - j - 1, centroidX][0]) {
                            y2_i = centroidY - j
                            break
                        }
                        j++
                    }

                    // Find minimum area rectangle
                    val rect = cnt?.let {
                        Imgproc.minAreaRect(MatOfPoint2f(*it.toArray()))
                    } ?: throw IllegalArgumentException("Contours cannot be null")

// Continue processin
                    val rect_Points = MatOfPoint2f()
                    Imgproc.boxPoints(rect, rect_Points)
// Get the 4 corner points of the rectangle

// Convert the points to integer values
                    val rectIntPoints = MatOfPoint()
                    rect_Points.convertTo(rectIntPoints, CvType.CV_32S)

                    val size = rect.size
                    var h1 = size.height.toFloat()
                    var w1 = size.width.toFloat()


                    var l1: Double
                    var b1: Double

                    if (h1 >= w1) {
                        l1 = h1 * mult2
                        b1 = w1 * mult1
                    } else {
                        l1 = w1 * mult2
                        b1 = h1 * mult1
                        // Swap h1 and w1
                        val temp = h1
                        h1 = w1
                        w1 = temp
                    }

// Get the bounding rectangle for the contour (cnt)
                    val boundingRect = Imgproc.boundingRect(cnt)
                    val x1 = boundingRect.x
                    val y1 = boundingRect.y
//                    val w2 = boundingRect.width
//                    val h2 = boundingRect.height

//                    val X_test = arrayOf(doubleArrayOf(l1))

                    b1 = (y1_i - y2_i) * mult1
                    breadths.add(b1)

                    val adjustedBox = rect_Points.clone() // Assuming rectPoints is a Mat
                    for (i in 0 until adjustedBox.rows()) {
                        val point = adjustedBox.row(i) // Get the i-th point as a Mat
                        point.put(0, 0, point.get(0, 0)[0] + startX) // Adjust x coordinates
                        point.put(0, 1, point.get(0, 1)[0] + startY) // Adjust y coordinates
                    }


// Draw bounding box on original image
                    // Convert adjustedBox (Mat) to MatOfPoint
                    val adjustedBoxPoints = MatOfPoint()
//                    adjustedBox.convertTo(adjustedBoxPoints, CvType.CV_32S) // Convert type if necessary

                    for (point in adjustedBoxPoints.toList()) {
//                        Imgproc.circle(drawonthis, Point(point.x, point.y), 5, Scalar(0.0, 255.0, 0.0), 10)
                        println("points : ${point.x}")
                    }


// Draw contours
                    Imgproc.drawContours(testMat, listOf(adjustedBoxPoints), -1, Scalar(0.0, 0.0, 0.0), 100)






                    val aspectRatio = h1.toFloat() / w1
                    ar.add(aspectRatio)
                    index.add(i)

                    val thresh1 = Mat()
                    val ret1 = Imgproc.threshold(imgBlur,thresh1, 207.0, 255.0, Imgproc.THRESH_BINARY) // imgBlur is the input Mat

                    val white1 = Core.countNonZero(thresh1)
                    val total1 = thresh1.total().toInt() // total() returns the total number of pixels
                    val whiteProportion1 = white1.toDouble() / total1

                    if (l1 < 6.61) {
                        riceType.add("Non-Basmati_$i")
                        Imgproc.putText(testMat, i.toString(),
                            Point(50.0,50.0),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0.0, 0.0, 255.0), 10)
                        Imgproc.putText(testMat, "NB",
                            Point((startX + x1 + 220).toDouble(), (startY + y1 + 100).toDouble()),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(0.0, 0.0, 255.0), 16)
//                        canvas.drawText(i.toString(), (startX + x1).toFloat(), (startY + y1 + 100).toFloat(), paint)
//                        canvas.drawText("NB", (startX + x1 + 220).toFloat(), (startY + y1 + 100).toFloat(), paint)
                        tnbas += 1


                        if (whiteProportion1 >= 0.02) {
                            chalkiness.add("chalky_$i")
                            Imgproc.putText(testMat, "C", Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)
//                            canvas.drawText("C", (startX + x1 + 500).toFloat(), (startY + y1 + 100).toFloat(), paint)

                        } else {
                            chalkiness.add("non-chalky_$i")
                            Imgproc.putText(testMat, "NC",
                                Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)
//                            canvas.drawText("NC", (startX + x1 + 500).toFloat(), (startY + y1 + 100).toFloat(), paint)

                        }
                    }else if (b1 > 2.01) {
                        riceType.add("Non-Basmati_$i")
                        Imgproc.putText(testMat, i.toString(),
                            Point((startX + x1).toDouble(), (startY + y1 + 100).toDouble()),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)
                        Imgproc.putText(testMat, "NB",
                            Point((startX + x1 + 220).toDouble(), (startY + y1 + 100).toDouble()),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 5.0, Scalar(255.0, 255.0, 0.0), 16)
////                        pos.add(filteredCoords[i])
//                        tnbas += 1
//                        canvas.drawText(i.toString(), (startX + x1).toFloat(), (startY + y1 + 100).toFloat(), paint)
//                        canvas.drawText("NB", (startX + x1 + 220).toFloat(), (startY + y1 + 100).toFloat(), paint)
                        tnbas += 1

                        if (whiteProportion1 >= 0.02) {
                            chalkiness.add("chalky_$i")
//                            canvas.drawText("C", (startX + x1 + 500).toFloat(), (startY + y1 + 100).toFloat(), paint)
                        } else {
                            chalkiness.add("non-chalky_$i")
//                            canvas.drawText("NC", (startX + x1 + 500).toFloat(), (startY + y1 + 100).toFloat(), paint)
                        }
                    } else if (aspectRatio < 3.5) {
                        riceType.add("Non-Basmati_$i")
//                        canvas.drawText(i.toString(), (startX + x1).toFloat(), (startY + y1 + 100).toFloat(), paint)
//                        canvas.drawText("NB", (startX + x1 + 220).toFloat(), (startY + y1 + 100).toFloat(), paint)
//                        pos.add(filteredCoords[i])
                        tnbas += 1

                        if (whiteProportion1 >= 0.02) {
                            chalkiness.add("chalky_$i")
//                            canvas.drawText("C", (startX + x1 + 500).toFloat(), (startY + y1 + 100).toFloat(), paint)
                        } else {
                            chalkiness.add("non-chalky_$i")
//                            canvas.drawText("NC", (startX + x1 + 500).toFloat(), (startY + y1 + 100).toFloat(), paint)
                        }
                    } else {
                        riceType.add("Basmati_$i")
//                        canvas.drawText(i.toString(), (startX + x1).toFloat(), (startY + y1 + 100).toFloat(), paint)
//                        canvas.drawText("B", (startX + x1 + 220).toFloat(), (startY + y1 + 100).toFloat(), paint)
//                        pos.add(filteredCoords[i])
                        tbas += 1

                        if (whiteProportion1 >= 0.02) {
                            chalkiness.add("chalky_$i")
//                            canvas.drawText("C", (startX + x1 + 500).toFloat(), (startY + y1 + 100).toFloat(), paint)
                        } else {
                            chalkiness.add("non-chalky_$i")
//                            canvas.drawText("NC", (startX + x1 + 500).toFloat(), (startY + y1 + 100).toFloat(), paint)
                        }
                    }

                }
            }
        }



        val testBitmap = Bitmap.createBitmap(drawonthis.cols(), drawonthis.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(drawonthis, testBitmap)


        imageView = findViewById(R.id.processed_image_view)
        imageView.setImageBitmap(testBitmap)
        imageView.invalidate() // Refresh ImageView

        return drawonthis
    }


    fun decodeModelOutput(output: Array<FloatArray>, confidenceThreshold: Float = 0.5f): Pair<List<FloatArray>, List<Float>> {
        // Assuming output shape is (batches, 4 + num_classes, num_candidate_detections)
        val xc = output[0] // Get x-coordinates
        val yc = output[1] // Get y-coordinates
        val w = output[2]  // Get width
        val h = output[3] // Get height
        val confidences = output[4] // Get confidences from output

        // Reshape the boxes from (4, num_candidate_detections) to (num_candidate_detections, 4)
        val boxes = mutableListOf<FloatArray>()
        for (i in xc.indices) {
            boxes.add(floatArrayOf(xc[i], yc[i], w[i], h[i]))
        }

        // Apply confidence threshold
        val filteredBoxes = mutableListOf<FloatArray>()
        val filteredConfidences = mutableListOf<Float>()

        for (i in confidences.indices) {
            if (confidences[i] > confidenceThreshold) {
                filteredBoxes.add(boxes[i])
                filteredConfidences.add(confidences[i])
            }
        }
        return Pair(filteredBoxes, filteredConfidences)
    }



    fun nonMaximumSuppression(boxes: List<Box>, confidences: List<Float>, scoreThreshold: Float, nmsThreshold: Float): List<Int> {
        val selectedIndices = mutableListOf<Int>()

        // Filter out boxes with low confidence scores
        val filteredIndices = confidences.indices.filter { confidences[it] > scoreThreshold }

        // Sort indices based on confidence scores (higher confidence first)
        val sortedIndices = filteredIndices.sortedByDescending { confidences[it] }

        var remainingIndices = sortedIndices
        while (remainingIndices.isNotEmpty()) {
            val currentIdx = remainingIndices[0]
            selectedIndices.add(currentIdx)

            val currentBox = boxes[currentIdx]
            remainingIndices = remainingIndices.drop(1).filter { idx ->
                computeIoU(currentBox, boxes[idx]) <= nmsThreshold
            }
        }

        return selectedIndices
    }


    fun computeIoU(boxA: Box, boxB: Box): Float {
        val xA = maxOf(boxA.x, boxB.x)
        val yA = maxOf(boxA.y, boxB.y)
        val xB = minOf(boxA.x + boxA.width, boxB.x + boxB.width)
        val yB = minOf(boxA.y + boxA.height, boxB.y + boxB.height)

        val intersectionArea = maxOf(0f, xB - xA) * maxOf(0f, yB - yA)
        val boxAArea = boxA.width * boxA.height
        val boxBArea = boxB.width * boxB.height

        return intersectionArea / (boxAArea + boxBArea - intersectionArea)
    }






    private fun saveImageToGallery(bitmap: Bitmap, title: String, description: String) {
        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "$title.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.DESCRIPTION, description)
            put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
        }

        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
        if (uri != null) {
            val outputStream = contentResolver.openOutputStream(uri)
            try {
                if (outputStream != null) {
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
                }
                outputStream?.close()
                Toast.makeText(this, "Image saved to gallery!", Toast.LENGTH_SHORT).show()
            } catch (e: IOException) {
                e.printStackTrace()
                Toast.makeText(this, "Failed to save image", Toast.LENGTH_SHORT).show()
            }
        }
    }

}