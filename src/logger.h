#ifndef LOGGER_H
#define LOGGER_H

#include "NvInfer.h"

using Severity = nvinfer1::ILogger::Severity;
class Logger : public nvinfer1::ILogger
{
public:
    //explicit Logger(Severity severity = Severity::kWARNING)
      //  : reportableSeverity(severity)
    //{}
    Logger(Severity severity = Severity::kWARNING)
    {
        const char* logLevel = std::getenv("TENSORRT_LOG_LEVEL");
        if (logLevel)
        {
            std::string level(logLevel);
            if (level == "INTERNAL_ERROR")
                reportableSeverity = Severity::kINTERNAL_ERROR;
            else if (level == "ERROR")
                reportableSeverity = Severity::kERROR;
            else if (level == "WARNING")
                reportableSeverity = Severity::kWARNING;
            else if (level == "INFO")
                reportableSeverity = Severity::kINFO;
            else if (level == "VERBOSE")
                reportableSeverity = Severity::kVERBOSE;
            else
                reportableSeverity = Severity::kWARNING; // default level
        }
        else
        {
            reportableSeverity = severity;
        }
    }

     void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= reportableSeverity)
        {
            switch (severity)
            {
                case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
                case Severity::kERROR: std::cerr << "ERROR: "; break;
                case Severity::kWARNING: std::cerr << "WARNING: "; break;
                case Severity::kINFO: std::cout << "INFO: "; break;
                case Severity::kVERBOSE: std::cout << "VERBOSE: "; break;
                default: std::cout << "UNKNOWN: "; break;
            }
            std::cout << msg << std::endl;
        }
    }


    void setReportableSeverity(Severity severity) noexcept
    {
        reportableSeverity = severity;
    }
    Severity getReportableSeverity() const
    {
        return reportableSeverity;
    }

private:

    Severity reportableSeverity;
};

#endif