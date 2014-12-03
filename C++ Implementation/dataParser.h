#ifndef data_parse
#define data_parse

const int MAX_CHARS_PER_LINE = 60;
const int MAX_TOKENS_PER_LINE = 23;
const int MAX_LINES = 8124;
const int MAX_TOKEN_TYPES = 10;
const int TEST_NUM = 10;
const int ANSWER_INDEX = 0;
const char* const DELIMITER = ",";

void getTestData(const char* file,
                 double trainingDataSet[]);

#endif
