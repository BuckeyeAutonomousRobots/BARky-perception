#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include "yolo_detector_cpp/tensorrt_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

class YoloDetectorNode : public rclcpp::Node {
public:
    YoloDetectorNode() : Node("yolo_detector_node") {
        declare_parameter<std::string>(
            "image_topic", "/stereo/left/image_raw");
        declare_parameter<std::string>(
            "engine_path",
            "/home/isaiahm/ros2_ws/src/yolo_detector_cpp/models/yolo26n.engine");
        declare_parameter<double>("confidence", 0.5);

        confidence_ = get_parameter("confidence").as_double();

        engine_ = std::make_unique<TensorRTEngine>(
            get_parameter("engine_path").as_string());

        annotated_pub_ =
            create_publisher<sensor_msgs::msg::Image>(
                "/yolo/annotated_image", rclcpp::QoS(1).reliable());

        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            get_parameter("image_topic").as_string(),
            rclcpp::QoS(1).reliable(),
            std::bind(&YoloDetectorNode::callback, this,
                      std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "YOLO TensorRT detector started");
    }

private:
    void callback(sensor_msgs::msg::Image::ConstSharedPtr msg) {
        try {
            // binds msg frame to a pointer for read only operations 
            // and avoids copying whole image 
            auto cv_ptr = cv_bridge::toCvShare(
                msg, sensor_msgs::image_encodings::BGR8);
            const cv::Mat& frame = cv_ptr->image;
            // scaling 
            const float scale = std::min(
                640.0f / frame.cols, 640.0f / frame.rows);

            // new frame resizing because yolo expects us to have a 640x640 image and we have 1280x720 
            // we resize it by adding padding to the image so the actual image will be 
            // 640x360 but the padding on the top and bottom think of a cinematic shot 
            // will black bars on the top and bottom will make it 640x640 
            const int new_w = std::round(frame.cols * scale);
            const int new_h = std::round(frame.rows * scale);
            const int pad_x = (640 - new_w) / 2;
            const int pad_y = (640 - new_h) / 2;
            // create two cv Mat objects resized and padded 
            cv::Mat resized, padded;
            // resize takes frame as input, resize as output, and Size as the designated size
            // Inter_area prevents aliasing artifacts from ppearing due to the shrinking 
            cv::resize(frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);

            // creates a 640x640 gray frame that its 8 bit unsigned(0-255) and 3 color channels 
            padded = cv::Mat(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
            // copies new scaled frame to gray frame so it is in the correct format with padding 
            resized.copyTo( padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

            //Open Cv natively works with BGR but YOLO wants RGB so we convert 
            // Generates a 4D CHW Tensor Blob (1x3x640x640) directly in memory
             cv::Mat input_blob = cv::dnn::blobFromImage(
            padded,               // 1. Input image source (640x640 BGR)
            1.0 / 255.0,          // 2. Normalization scale factor (transforms 0-255 to 0.0-1.0)
            cv::Size(640, 640),   // 3. Confirm target spatial dimensions
            cv::Scalar(0, 0, 0),  // 4. Mean subtraction (none needed for standard YOLO)
            true,                 // 5. Swap R and B channels (Handles BGR -> RGB swap on the fly)
            false,                // 6. Center Crop (false, we handled padding manually)
            CV_32F                // 7. Output target data type (32-bit float)
            );
         
            
            

            

            std::vector<float> output;
            if (!engine_->infer(input_blob.ptr<float>(), output)) {
                RCLCPP_WARN(get_logger(), "Inference failed");
                return;
            }

            // Output: 300 detections x 6 values
            // [x1, y1, x2, y2, confidence, class_id]
            // det[0] = left edge of box , det[1] = Top edge of box, det[2] = right edge of box 
            // det[3] = bottom edge of box , det[4] = confidence score , det[5] = class id 
            cv::Mat annotated = frame.clone();
            int person_count = 0;
           
            for (int i = 0; i < 300; ++i) {
                // every iteration is a different set of detections 
                const float* det = &output[i * 6];
                
                // makes sure that that score is finite and above designated confidence score 
                // makes sure that if the class id is rounded and its still not zero
                // because we are looking for class id 0 for person 
                if (!std::isfinite(det[4]) ||
                    det[4] < confidence_ ||
                    !std::isfinite(det[5]) ||
                    std::round(det[5]) != 0) {
                    continue;
                }
                // checking if the det[0:4] is finite 
                if (!std::isfinite(det[0]) ||
                    !std::isfinite(det[1]) ||
                    !std::isfinite(det[2]) ||
                    !std::isfinite(det[3])) {
                    continue;
                }
                // making sure values are within frame limits 
                int x1 = std::clamp( static_cast<int>((det[0] - pad_x) / scale), 0, frame.cols - 1);
                int y1 = std::clamp( static_cast<int>((det[1] - pad_y) / scale), 0, frame.rows - 1);
                int x2 = std::clamp( static_cast<int>((det[2] - pad_x) / scale), 0, frame.cols - 1);
                int y2 = std::clamp( static_cast<int>((det[3] - pad_y) / scale), 0, frame.rows - 1);

                if (x2 <= x1 || y2 <= y1) continue;
                // draw bounding box 
                cv::rectangle(
                    annotated, cv::Point(x1, y1),
                    cv::Point(x2, y2),
                    cv::Scalar(0, 255, 0), 2);
                // draw text box 
                cv::putText(
                    annotated,
                    cv::format("Person %.2f", det[4]),
                    cv::Point(x1, std::max(20, y1 - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);

                ++person_count;
            }

            // Preserve the original camera timestamp.
            auto annotated_msg = cv_bridge::CvImage(
                msg->header,
                sensor_msgs::image_encodings::BGR8,
                annotated).toImageMsg();

            annotated_pub_->publish(*annotated_msg);

            ++frame_count_;
            const auto now = std::chrono::steady_clock::now();
            const double seconds =
                std::chrono::duration<double>(now - last_log_).count();
            // post to logger if its been longer thn 5 seconds 
            if (seconds >= 5.0) {
                RCLCPP_INFO(
                    get_logger(),
                    "YOLO FPS: %.2f | People: %d",
                    frame_count_ / seconds, person_count);
                frame_count_ = 0;
                last_log_ = now;
            }

        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "YOLO error: %s", e.what());
        }
    }

    std::unique_ptr<TensorRTEngine> engine_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_pub_;
    double confidence_ = 0.5;
    int frame_count_ = 0;
    std::chrono::steady_clock::time_point last_log_ =
        std::chrono::steady_clock::now();
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<YoloDetectorNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        std::cerr << "Fatal YOLO error: " << e.what() << '\n';
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}