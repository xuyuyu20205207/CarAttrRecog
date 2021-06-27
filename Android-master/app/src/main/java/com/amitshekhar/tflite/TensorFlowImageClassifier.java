package com.amitshekhar.tflite;

import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Created by amitshekhar on 17/03/18.
 */

public class TensorFlowImageClassifier implements Classifier {

    private static final int MAX_RESULTS = 3;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;
    private static final float THRESHOLD = 0.1f;

//    private static final int IMAGE_MEAN = 128;
//    private static final float IMAGE_STD = 128.0f;
    private static final float IMAGE_MEAN_1 = 0.485f;
    private static final float IMAGE_STD_1 = 0.229f;
    private static final float IMAGE_MEAN_2 = 0.456f;
    private static final float IMAGE_STD_2 = 0.224f;
    private static final float IMAGE_MEAN_3 = 0.406f;
    private static final float IMAGE_STD_3 = 0.225f;


    private Interpreter interpreter;
    private int inputSize;
    private List<String> labelList;
    private boolean quant;

    private TensorFlowImageClassifier() {

    }

    static Classifier create(AssetManager assetManager,
                             String modelPath,
                             String labelPath,
                             int inputSize,
                             boolean quant) throws IOException {

        TensorFlowImageClassifier classifier = new TensorFlowImageClassifier();
        classifier.interpreter = new Interpreter(classifier.loadModelFile(assetManager,
                modelPath), new Interpreter.Options());
        classifier.labelList = classifier.loadLabelList(assetManager, labelPath);
        classifier.inputSize = inputSize;
        classifier.quant = quant;

        return classifier;
    }

    // error for car attr

    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        if(quant){
            //1*196
            byte[][] result = new byte[1][labelList.size()];
            interpreter.run(byteBuffer, result);
            return getSortedResultByte(result);
        } else {
            float [][] result = new float[1][labelList.size()];
            interpreter.run(byteBuffer, result);
            return getSortedResultFloat(result);
        }

    }

    public String[] recognizeImage(Bitmap bitmap,int type) {
        FloatBuffer buffer = convertBitmapToFloatBuffer(bitmap);
        float [][] result = new float[1][labelList.size()];
        interpreter.run(buffer, result);
        return getSortedResultFloat(result,type);
    }

    @Override
    public void close() {
        interpreter.close();
        interpreter = null;
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
//        List<String> labelList = new ArrayList<>();
//        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
//        String line;
//        while ((line = reader.readLine()) != null) {
//            labelList.add(line);
//        }
//        reader.close();
//        return labelList;
        List<String> labelList = new ArrayList<>(196);
        for(int i=0;i<196;++i){
            labelList.add("");
        }

        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] strs=line.split(" ");
            int id=Integer.parseInt(strs[1]);
            labelList.set(id, strs[2]);
        }
        reader.close();
        return labelList;
    }


    private float[] getRGBFromBMP(Bitmap bmp) {

        int w = bmp.getWidth();
        int h = bmp.getHeight();

        float[] pixels = new float[w * h * 3]; // Allocate for RGB

        int k = 0;

        for (int x = 0; x < h; x++) {
            for (int y = 0; y < w; y++) {
                int color = bmp.getPixel(y, x);
                pixels[k * 3] = (((float) Color.red(color))-IMAGE_MEAN_1)/IMAGE_STD_1;
                pixels[k * 3 + 1] = (((float) Color.green(color))-IMAGE_MEAN_2)/IMAGE_STD_2;
                pixels[k * 3 + 2] = (((float) Color.blue(color))-IMAGE_MEAN_3)/IMAGE_STD_3;

               // Log.i("x,y",pixels[k * 3]+" "+pixels[k * 3 + 1]+" "+pixels[k * 3 + 2]);
                k++;
            }
        }



        return pixels;
    }

    private FloatBuffer convertBitmapToFloatBuffer(Bitmap bitmap){
        FloatBuffer floatBuffer;
//        floatBuffer=FloatBuffer.allocate(inputSize*inputSize*PIXEL_SIZE);
        float[] rgb=getRGBFromBMP(bitmap);
        floatBuffer=FloatBuffer.wrap(rgb);
        return floatBuffer;
    }

    //error for car_attr recognize
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;

        if(quant) {
            //[none,width,height,channel]
            byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        } else {
            byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        }

        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                if(quant){
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));  // high 16
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    byteBuffer.put((byte) (val & 0xFF));
                } else {
//                    byteBuffer.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
//                    byteBuffer.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
//                    byteBuffer.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                }

            }
        }

        return byteBuffer;
    }

    @SuppressLint("DefaultLocale")
    private List<Recognition> getSortedResultByte(byte[][] labelProbArray) {

        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int i = 0; i < labelList.size(); ++i) {
            float confidence = (labelProbArray[0][i] & 0xff) / 255.0f;
            if (confidence > THRESHOLD) {
                pq.add(new Recognition("" + i,
                        labelList.size() > i ? labelList.get(i) : "unknown",
                        confidence, quant));
            }
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }

    @SuppressLint("DefaultLocale")
    private List<Recognition> getSortedResultFloat(float[][] labelProbArray) {

        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int i = 0; i < labelList.size(); ++i) {
            float confidence = labelProbArray[0][i];
            if (confidence > THRESHOLD) {
                pq.add(new Recognition("" + i,
                        labelList.size() > i ? labelList.get(i) : "unknown",
                        confidence, quant));
            }
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }

    private String[] getSortedResultFloat(float[][] labelProbArray,int type) {

        float maxVal=0;
        int maxIndex=0;
        for(int i=0;i<labelProbArray[0].length;++i){
            if(maxVal<labelProbArray[0][i]){
                maxVal=labelProbArray[0][i];
                maxIndex=i;
            }
        }
        return new String[]{Integer.valueOf(maxIndex).toString(),labelList.get(maxIndex)};
    }

    private String[] getSortedResultByte(byte[][] labelProbArray,int type) {

        byte maxVal=0;
        int maxIndex=0;
        for(int i=0;i<labelProbArray[0].length;++i){
            if(maxVal<labelProbArray[0][i]){
                maxVal=labelProbArray[0][i];
                maxIndex=i;
            }
        }
        return new String[]{Integer.valueOf(maxIndex).toString(),labelList.get(maxIndex)};
    }

}
