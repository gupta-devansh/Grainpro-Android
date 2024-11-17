package com.example.basmati

import kotlinx.coroutines.*
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.RectF
import android.graphics.Typeface

import android.icu.number.Scale
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.text.Spannable
import android.text.SpannableString
import android.text.style.RelativeSizeSpan
import android.text.style.ReplacementSpan
import android.text.style.StyleSpan
import android.util.Log
import android.view.MotionEvent
import android.widget.Button
import android.widget.ImageView
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.basmati.ml.Best5Float32
import com.github.mikephil.charting.charts.BarChart
import com.github.mikephil.charting.components.AxisBase
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.BarData
import com.github.mikephil.charting.data.BarDataSet
import com.github.mikephil.charting.data.BarEntry
import com.github.mikephil.charting.formatter.ValueFormatter
import com.github.mikephil.charting.utils.ColorTemplate
import com.ortiz.touchview.TouchImageView
//import org.apache.poi.xssf.usermodel.XSSFWorkbook
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
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.FileWriter
import java.io.IOException
import java.io.OutputStream
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

class RoundedBackgroundOutlineSpan(
    private val outlineColor: Int,
    private val textColor: Int,
    private val strokeWidth: Float,
    private val padding: Float,
    private val lineSpacing: Float // New parameter for line spacing
) : ReplacementSpan() {

    override fun getSize(
        paint: Paint,
        text: CharSequence,
        start: Int,
        end: Int,
        fm: Paint.FontMetricsInt?
    ): Int {
        return (paint.measureText(text.subSequence(start, end).toString()) + padding * 2).toInt()
    }

    override fun draw(
        canvas: Canvas,
        text: CharSequence,
        start: Int,
        end: Int,
        x: Float,
        top: Int,
        y: Int,
        bottom: Int,
        paint: Paint
    ) {
        val textString = text.subSequence(start, end).toString()
        val width = paint.measureText(textString)

        // Set the outline color and style
        val outlinePaint = Paint().apply {
            color = outlineColor
            style = Paint.Style.STROKE
            strokeWidth = this@RoundedBackgroundOutlineSpan.strokeWidth
        }

        // Set the background color as transparent
        val rect = RectF(x - padding, top.toFloat(), x + width + padding, bottom.toFloat())

        // Draw the outline with rounded corners
        canvas.drawRoundRect(rect, 15f, 15f, outlinePaint)

        // Draw the text with increased line spacing
        paint.color = textColor
        canvas.drawText(textString, x, y.toFloat() + lineSpacing, paint) // Adjusting Y coordinate
    }
}







class MainActivity : AppCompatActivity() {

    private lateinit var mlModelHandler: MLModelHandler
    lateinit var imageView: ImageView
    lateinit var final : TouchImageView
    lateinit var scrollview: ScrollView
    lateinit var original: Bitmap
    lateinit var button : Button
//    lateinit var model: Best5Float32

//    var imageProcessor = ImageProcessor.Builder().add(ResizeOp(640,640,ResizeOp.ResizeMethod.BILINEAR)).build()

    data class Box(val x: Float, val y: Float, val width: Float, val height: Float)

//    data class Detection(val box: Box, val confidence: Float)


    override fun onCreate(savedInstanceState: Bundle?) {
        System.loadLibrary("opencv_java4") // or "opencv_java" based on your version

        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        val barChart = findViewById<BarChart>(R.id.bar_chart)
        barChart.setNoDataText("")  // Set an empty string to hide the default message
        barChart.setNoDataTextColor(Color.TRANSPARENT)

        mlModelHandler = MLModelHandler(this)


        val intent = Intent()
        intent.setType("image/*")
        intent.setAction(Intent.ACTION_GET_CONTENT)

//        model = Best5Float32.newInstance(this)

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

            val processedImageTextView: TextView = findViewById(R.id.processed_image_text)
            val desiredText = "...PROCESSING..."
            processedImageTextView.text = desiredText

            Handler(Looper.getMainLooper()).postDelayed({
                // Simulate executing a function to get text2
                val sampleop = get_prediction()
                val liness = sampleop.split("\n")

// Call the function to justify the text
                val justifiedText = justifyTextToFullWidth(processedImageTextView, liness)
                val spannableString = SpannableString(justifiedText)

                val totalIndex = justifiedText.indexOf("Total Rice Grains detected")
                val adulterationIndex = justifiedText.indexOf("Adulteration")

// Apply size 1.2f for the part related to total rice grains and basmati/non-basmati counts
                val outlineColors = listOf(Color.DKGRAY, Color.DKGRAY, Color.DKGRAY, Color.RED, Color.RED, Color.RED, Color.RED,Color.RED)
                val textColor = Color.BLACK  // Text color for all lines
                val strokeWidth = 5f  // Width of the outline stroke
                val padding = 0f  // Padding around the boxes
                val lineSpacing = 5f // Increase this value for more spacing between lines

// Split the text into individual lines
                val lines = justifiedText.split("\n")

                var currentIndex = 0

                for ((i, line) in lines.withIndex()) {
                    val lineLength = line.length
                    val nextIndex = currentIndex + lineLength

                    // Apply the outline span with different outline colors and line spacing
                    spannableString.setSpan(
                        RoundedBackgroundOutlineSpan(outlineColors[i % outlineColors.size], textColor, strokeWidth, padding, lineSpacing),
                        currentIndex,
                        nextIndex,
                        Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                    )

                    // Move to the next line
                    currentIndex = nextIndex + 1  // +1 accounts for the newline character
                }

                spannableString.setSpan(
                    RelativeSizeSpan(1.5f),
                    totalIndex,
                    adulterationIndex,  // Apply from the start until the "Adulteration" part
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )

// Apply size 1.5f and bold style for the adulteration and length/breadth part
                spannableString.setSpan(
                    RelativeSizeSpan(1.5f),
                    adulterationIndex,
                    spannableString.length,  // From "Adulteration" till the end of the string
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )

                spannableString.setSpan(
                    StyleSpan(Typeface.BOLD),
                    adulterationIndex,
                    spannableString.length,// From "Adulteration" till the end of the string
                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                )
                processedImageTextView.text = spannableString

            }, 10)



        }
    }

//    override fun onDestroy() {
//        super.onDestroy()
//        model.close()
//    }






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

    fun get_prediction(): String {

        var adjustedBitmap = adjustImageOrientation(original)
        if (adjustedBitmap == null) {
            Log.e("Error", "Adjusted bitmap is null")
            return "Adjusted bitmap is null"
        }
        var matImage = Mat()
        Utils.bitmapToMat(adjustedBitmap,matImage)
        if (matImage.empty()) {
            Log.e("Error", "Mat image is empty after conversion from bitmap")
            return "Mat image is empty after conversion from bitmap"
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

        var sop = "test"

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
            val finalSize = Size(8741.0, 10000.0)
            Imgproc.resize(warpedImage, warpedImage, finalSize, 0.0, 0.0, Imgproc.INTER_CUBIC)


            var processedBitmap = Bitmap.createBitmap(warpedImage.cols(), warpedImage.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(warpedImage, processedBitmap)




            try {
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

                val imageWidth = 8741f
                val imageHeight = 10000f

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
                    val endXInt = Math.min(8741,endX.toInt())
                    val endYInt = Math.min(10000,endY.toInt())

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

                rectList.sortBy { (it.x * Int.MAX_VALUE) + it.y }



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
                val breadths = mutableListOf<Float>()
                val index = mutableListOf<Int>()
                var tbas = 0
                var tnbas = 0
                var chalky = 0

                // Scaling multipliers
                val mult1 = 60.toFloat() / 10000
                val mult2 = 52.45.toFloat()/ 8741

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
                    val totalPixels = thresh.total().toInt()
                    val whiteProportion = whitePixels.toDouble() /totalPixels.toDouble()

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


                            var l1: Float
                            var b1: Float


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
                            lengths.add(l1)
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
                            Imgproc.drawContours(inputImageMat, listOf(matOfPoint), 0, Scalar(0.0, 255.0, 0.0), 15)

                            val aspectRatio = h1.toFloat() / w1
                            ar.add(aspectRatio)
                            index.add(i)

                            val thresh1 = Mat()
                            val ret1 = Imgproc.threshold(imgBlur,thresh1, 207.0, 255.0, Imgproc.THRESH_BINARY) // imgBlur is the input Mat

                            val white1 = Core.countNonZero(thresh1)
                            // Get total number of pixels
                            val total1 = thresh1.total().toInt() // total() returns the total number of pixels
                            val whiteProportion1 = white1.toDouble() / total1.toDouble()

                            if (l1 < 6.61) {
                                riceType.add("Non-Basmati_$i")
                                Imgproc.putText(inputImageMat, i.toString(),
                                    Point((startX+x1).toDouble(),(startY+y1+100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(255.0, 255.0, 0.0), 20)
                                Imgproc.putText(inputImageMat, " NB",
                                    Point((startX + x1 + 220).toDouble(), (startY + y1 + 100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(255.0, 255.0, 0.0), 20)
                                tnbas += 1


                                if (whiteProportion1 >= 0.02) {
                                    chalky+=1
                                    chalkiness.add("chalky_$i")
                                    Imgproc.putText(inputImageMat, " C", Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(255.0, 255.0, 0.0), 20)

                                } else {
                                    chalkiness.add("non-chalky_$i")
                                    Imgproc.putText(inputImageMat, " NC",
                                        Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(255.0, 255.0, 0.0), 20)
                                }
                            }else if (b1 > 2.01) {
                                riceType.add("Non-Basmati_$i")
                                Imgproc.putText(inputImageMat, i.toString(),
                                    Point((startX+x1).toDouble(),(startY+y1+100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(255.0, 255.0, 0.0), 20)
                                Imgproc.putText(inputImageMat, " NB",
                                    Point((startX + x1 + 220).toDouble(), (startY + y1 + 100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(255.0, 255.0, 0.0), 20)
                                tnbas += 1


                                if (whiteProportion1 >= 0.02) {
                                    chalky+=1
                                    chalkiness.add("chalky_$i")
                                    Imgproc.putText(inputImageMat, " C", Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(255.0, 255.0, 0.0), 20)

                                } else {
                                    chalkiness.add("non-chalky_$i")
                                    Imgproc.putText(inputImageMat, " NC",
                                        Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(255.0, 255.0, 0.0), 20)
                                }
                            } else if (aspectRatio < 3.5) {
                                riceType.add("Non-Basmati_$i")
                                Imgproc.putText(inputImageMat, i.toString(),
                                    Point((startX+x1).toDouble(),(startY+y1+100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(255.0, 255.0, 0.0), 20)
                                Imgproc.putText(inputImageMat, " NB",
                                    Point((startX + x1 + 220).toDouble(), (startY + y1 + 100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(255.0, 255.0, 0.0), 20)
                                tnbas += 1


                                if (whiteProportion1 >= 0.02) {
                                    chalky+=1
                                    chalkiness.add("chalky_$i")
                                    Imgproc.putText(inputImageMat, " C", Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(255.0, 255.0, 0.0), 20)

                                } else {
                                    chalkiness.add("non-chalky_$i")
                                    Imgproc.putText(inputImageMat, " NC",
                                        Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(255.0, 255.0, 0.0), 20)
                                }
                            } else {
                                riceType.add("Basmati_$i")
                                Imgproc.putText(inputImageMat, i.toString(),
                                    Point((startX+x1).toDouble(),(startY+y1+100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(0.0, 0.0, 255.0), 20)
                                Imgproc.putText(inputImageMat, " B",
                                    Point((startX + x1 + 220).toDouble(), (startY + y1 + 100).toDouble()),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(0.0, 0.0, 255.0), 20)
                                tbas += 1


                                if (whiteProportion1 >= 0.02) {
                                    chalky+=1
                                    chalkiness.add("chalky_$i")
                                    Imgproc.putText(inputImageMat, " C", Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(0.0, 0.0, 255.0), 20)

                                } else {
                                    chalkiness.add("non-chalky_$i")
                                    Imgproc.putText(inputImageMat, " NC",
                                        Point((startX + x1 + 500).toDouble(), (startY + y1 + 100).toDouble()),
                                        Imgproc.FONT_HERSHEY_SIMPLEX, 7.0, Scalar(0.0, 0.0, 255.0), 20)
                                }
                            }

                        }
                    }


                }



                val resultbitmap = Bitmap.createBitmap(inputImageMat.cols(),inputImageMat.rows(),Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(inputImageMat,resultbitmap)

                imageView = findViewById(R.id.processed_image_view)
                scrollview = findViewById(R.id.scroll_view)



                // Get display dimensions
                val displayMetrics = Resources.getSystem().displayMetrics
                val screenWidth = displayMetrics.widthPixels
                val screenHeight = displayMetrics.heightPixels

// Resize the bitmap
                val resizedBitmap = resizeBitmap(resultbitmap, screenWidth, screenHeight)




                imageView.setImageBitmap(resizedBitmap)



                saveImageToGallery(resultbitmap, "final Image", "This is a processed image")
                imageView.setOnTouchListener { v, event ->
                    when (event.action) {
                        MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                            // Disable ScrollView's interception of touch events
                            scrollview.requestDisallowInterceptTouchEvent(true)
                        }
                        MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                            // Re-enable ScrollView's interception when touch ends
                            scrollview.requestDisallowInterceptTouchEvent(false)
                        }
                    }
                    // Pass the event to the TouchImageView for further handling (zooming, etc.)
                    v.onTouchEvent(event)
                }

                // Set the desired text
                val adulteration = tnbas * 100.0 / (tbas + tnbas)
                val roundedAdulteration = String.format("%.2f", adulteration)

                val avlen = lengths.average()
                val avbr = breadths.average()
                val avar = avlen/avbr
                val chalkyp = (chalky*100.0)/(tnbas+tbas)
                val chalkyps = String.format("%.2f",chalkyp)
                val ravlen = String.format("%.3f",avlen)
                val ravbr = String.format("%.3f",avbr)
                val ravar = String.format("%.3f",avar)

//                val processedImageTextView: TextView = findViewById(R.id.processed_image_text)
                val desiredText = "Total Rice Grains detected : ${tbas+tnbas}\nNo. of Basmati grains : ${tbas}\nNo. of Non-Basmati grains : ${tnbas}\nAdulteration : ${roundedAdulteration}%\nAverage length : ${ravlen} mm\nAverage breadth : ${ravbr} mm\n" +
                        "Average aspect ratio : ${ravar}\nChalky percentage : ${chalkyps}%"

//                processedImageTextView.text = desiredText

                val barChart = findViewById<BarChart>(R.id.bar_chart)
                val barChartTitle = findViewById<TextView>(R.id.bar_chart_title)

// Optional: Change the title dynamically if needed
                barChartTitle.text = "Length of Grains"

//                barChart.visibility = view.VISIBLE
                val binSize = 0.5f
                val (histogramData, binStarts) = createHistogram(lengths, binSize)

                val barEntries = histogramData.mapIndexed { index, frequency ->
                    BarEntry(index.toFloat(), frequency)
                }

// Set up the chart with bar entries and bin start values
                val barDataSet = BarDataSet(barEntries, "Histogram")
                barDataSet.colors = listOf(Color.BLUE)
                val barData = BarData(barDataSet)
                barChart.data = barData

// Customize the chart
                barChart.description.isEnabled = false
                barChart.animateY(1000)

                val legend = barChart.legend
                legend.isEnabled = false

                val xAxis = barChart.xAxis
                xAxis.position = XAxis.XAxisPosition.BOTTOM
                xAxis.setDrawGridLines(false)

// Update the X-axis value formatter to display bin start values
                xAxis.valueFormatter = object : ValueFormatter() {
                    override fun getAxisLabel(value: Float, axis: AxisBase?): String {
                        val index = value.toInt()
                        return if (index in binStarts.indices) {
                            String.format("%.1f", binStarts[index]) // Display the bin start value
                        } else {
                            ""
                        }
                    }
                }

                val leftAxis = barChart.axisLeft
                leftAxis.isEnabled = false
                val rightAxis = barChart.axisRight
                rightAxis.isEnabled = false

// Remove the grid lines from the Y-axis
                leftAxis.setDrawGridLines(false)
                rightAxis.setDrawGridLines(false)


                sop = desiredText

                saveCsvToDocuments(index, lengths, breadths)




            } catch (e: IOException) {
                e.printStackTrace()
                sop = "Image Error, upload a different Image"
            }

        }

        return sop
    }


    fun resizeBitmap(bitmap: Bitmap, maxWidth: Int, maxHeight: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        val aspectRatio = width.toFloat() / height.toFloat()

        val newWidth = if (width > maxWidth) maxWidth else width
        val newHeight = (newWidth / aspectRatio).toInt().coerceAtMost(maxHeight)

        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }



    fun saveCsvToDocuments(list1: List<Int>, list2: List<Float>, list3: List<Float>) {
        val fileName = "data.csv"
        val csvHeader = "Column1,Column2,Column3\n"

        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
            put(MediaStore.MediaColumns.MIME_TYPE, "text/csv")
            put(MediaStore.MediaColumns.RELATIVE_PATH, "Documents/")
        }

        val uri = contentResolver.insert(MediaStore.Files.getContentUri("external"), contentValues)

        uri?.let {
            try {
                val outputStream: OutputStream? = contentResolver.openOutputStream(it)
                outputStream?.let { stream ->
                    stream.write(csvHeader.toByteArray())
                    for (i in list1.indices) {
                        val row = "${list1[i]},${list2[i]},${list3[i]}\n"
                        stream.write(row.toByteArray())
                    }
                    stream.flush()
                    stream.close()
                }
                Toast.makeText(this, "CSV file saved to Documents", Toast.LENGTH_LONG).show()
            } catch (e: IOException) {
                e.printStackTrace()
                Toast.makeText(this, "Error saving CSV file", Toast.LENGTH_SHORT).show()
            }
        }
    }










    fun justifyTextToFullWidth(textView: TextView, lines: List<String>): String {
        val paint = Paint()
        paint.textSize = textView.textSize

        val screenWidth = Resources.getSystem().displayMetrics.widthPixels
        val spaceWidth = paint.measureText(" ")

        val justifiedText = StringBuilder()

        for (line in lines) {
            val textWidth = paint.measureText(line)
            val remainingSpace = screenWidth - textWidth
            val numberOfSpaces = (remainingSpace / spaceWidth).toInt()

            // Add the line with spaces at the end
            justifiedText.append(line).append(" ".repeat(numberOfSpaces)).append("\n")
        }

        return justifiedText.toString()
    }

    fun createHistogram(data: MutableList<Float>, binSize: Float): Pair<MutableList<Float>, List<Float>> {
        val maxValue = data.maxOrNull() ?: 0f
        val minValue = data.minOrNull() ?: 0f
        val numberOfBins = ((maxValue - minValue) / binSize).toInt() + 1

        // Initialize list to store bin frequencies
        val bins = MutableList(numberOfBins) { 0f }
        val binStarts = mutableListOf<Float>() // List to store bin start values

        // Populate bins based on the data values
        for (value in data) {
            val binIndex = ((value - minValue) / binSize).toInt()
            println(binIndex)
            if (binIndex in bins.indices) {
                bins[binIndex] = bins[binIndex] + 1
            }
        }

        // Populate the binStarts list with the starting value of each bin
        for (i in 0 until numberOfBins) {
            val binStart = minValue + i * binSize
            binStarts.add(binStart)
            println(binStart)
        }

        return Pair(bins, binStarts)
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