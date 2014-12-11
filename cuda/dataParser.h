#ifndef data_parse
#define data_parse

const int MAX_TOKEN_TYPES = 10;
const char* const DELIMITER = ",";

void getTestData(const char* file,
                 double *trainingDataSet,
                 int lineSize,
                 int isImage);

#endif
