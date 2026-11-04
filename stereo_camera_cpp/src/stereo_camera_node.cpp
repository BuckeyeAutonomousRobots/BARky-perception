#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/msg/header.hpp> 

#include <opencv2/opencv.hpp>

#include <chrono> 
#include <memory>
#include <string>   

using namespace std::chrono_literals;

class StereoCameraNode : public rclcpp::Node {
public:
    StereoCameraNode() : Node("stereo_camera_node"), frame_count_(0) {
        // 1. Declare parameters with correct types
        this->declare_parameter<std::string>("device", "/dev/video0");
        this->declare_parameter<int>("width", 2560);
        this->declare_parameter<int>("height", 720);
        this->declare_parameter<int>("fps", 30);

        // 2. Retrieve parameters with matching assignments
        std::string device = this->get_parameter("device").as_string();
        int width = this->get_parameter("width").as_int();
        int height = this->get_parameter("height").as_int();
        int fps = this->get_parameter("fps").as_int();

        // 3. Setup publishers with correct variable names and matching types
        auto qos = rclcpp::QoS(1).reliable();
        left_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/left/image_raw", qos);
        right_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/right/image_raw", qos);
        combined_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/stereo/combined/image_raw", qos);
        
        // 4. Initialize OpenCV video capture
        cap_.open(device, cv::CAP_V4L2);
        if(!cap_.isOpened()) {
            throw std::runtime_error("Could not open camera: " + device);
        }

        cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap_.set(cv::CAP_PROP_FPS, fps);

        RCLCPP_INFO(
            this->get_logger(), 
            "Camera opened: %.0f x %.0f", 
            cap_.get(cv::CAP_PROP_FRAME_WIDTH), 
            cap_.get(cv::CAP_PROP_FRAME_HEIGHT)
        );

        // 5. Initialize loop components
        fps_start_ = std::chrono::steady_clock::now();

        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / fps), 
            std::bind(&StereoCameraNode::capture_frame, this)
        );
    }

    ~StereoCameraNode() {
        if (cap_.isOpened()) {
            cap_.release();
        }
    }

private: 
    cv::VideoCapture cap_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr right_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr combined_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    int frame_count_;

    std::chrono::steady_clock::time_point fps_start_;

    void capture_frame() {
        auto t0 = std::chrono::steady_clock::now();

        cv::Mat frame;
        bool success = cap_.read(frame);
        auto t1 = std::chrono::steady_clock::now();

          if (!success || frame.empty()) {
        RCLCPP_WARN(
            this->get_logger(),
            "Failed to capture frame"
        );
        return;
        }


        if (frame.empty() || frame.cols % 2 != 0) {
            RCLCPP_WARN(this->get_logger(), "Invalid stereo frame layout");
            return;
        }

        int midpoint = frame.cols / 2;
        cv::Mat left = frame(cv::Rect(0, 0, midpoint, frame.rows));
        cv::Mat right = frame(cv::Rect(midpoint, 0, midpoint, frame.rows));

        auto stamp = this->now();

        std_msgs::msg::Header left_header;
        left_header.stamp = stamp;
        left_header.frame_id = "stereo_left";

        std_msgs::msg::Header right_header;
        right_header.stamp = stamp;
        right_header.frame_id = "stereo_right";

        std_msgs::msg::Header combined_header;
        combined_header.stamp = stamp;
        combined_header.frame_id = "stereo_combined";

        auto left_msg = cv_bridge::CvImage(
            left_header, 
            sensor_msgs::image_encodings::BGR8, 
            left
        ).toImageMsg();

        auto right_msg = cv_bridge::CvImage(
            right_header,
            sensor_msgs::image_encodings::BGR8,
            right
        ).toImageMsg();

        auto combined_msg = cv_bridge::CvImage(
            combined_header, 
            sensor_msgs::image_encodings::BGR8,
            frame
        ).toImageMsg();
        
       if (combined_pub_->get_subscription_count() > 0 ||
        combined_pub_->get_intra_process_subscription_count() > 0) {
            auto combined_msg = cv_bridge::CvImage(
            combined_header,
            sensor_msgs::image_encodings::BGR8,
            frame
            ).toImageMsg();

             combined_pub_->publish(*combined_msg);
        }

        left_pub_->publish(*left_msg);
        right_pub_->publish(*right_msg);
        combined_pub_->publish(*combined_msg);
        

        frame_count_++;

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - fps_start_).count();

        if(elapsed >= 5.0) {
            RCLCPP_INFO(
                this->get_logger(), 
                "Publishing FPS: %.2f",
                frame_count_ / elapsed
            );

            frame_count_ = 0;
            fps_start_ = now;
        }
        auto t2 = std::chrono::steady_clock::now();

        double capture_ms = std::chrono::duration<double, std::milli>(
            t1-t0
        ).count();

        double processing_ms = std::chrono::duration<double, std::milli>(
            t2-t1
        ).count();

        static int diagnostic_count = 0;

        if (++diagnostic_count >= 30) {
            RCLCPP_INFO(
                this->get_logger(),
                "Capture %.2f ms| Processing + Publish: %.2f ms",
                capture_ms,
                processing_ms
            );

            diagnostic_count = 0;
        }
        

    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<StereoCameraNode>();
        rclcpp::spin(node);
    }
    catch (const std::exception &e) {
        RCLCPP_ERROR(
            rclcpp::get_logger("stereo_camera"),
            "Error: %s",
            e.what()
        );
    }
    rclcpp::shutdown();
    return 0;
}