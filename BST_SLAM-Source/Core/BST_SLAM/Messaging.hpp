#pragma once
#include <opencv2/opencv.hpp>

namespace BST_SLAM {
    typedef struct {
        float Tx, Ty, Tz;
        float Rx, Ry, Rz, Rw;
    } MessageData;

    inline void SendMessage(const MessageData& Message) {
        std::stringstream PipeOut;
        PipeOut << Message.Tx << " " << Message.Ty << " " << Message.Tz << " "
                << Message.Rx << " " << Message.Ry << " " << Message.Rz << " " << Message.Rw;
        puts(PipeOut.str().c_str());
        std::cout << PipeOut.str().c_str() << std::endl;
    }

    inline void SendMessage(const cv::Vec3f& T, const cv::Vec4f& R) {
        MessageData Message;
        Message.Tx = T[0];
        Message.Ty = T[1];
        Message.Tz = T[2];
        Message.Rx = R[0];
        Message.Ry = R[1];
        Message.Rz = R[2];
        Message.Rw = R[3];
        SendMessage(Message);
    }
};