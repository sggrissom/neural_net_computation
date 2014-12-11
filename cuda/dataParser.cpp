#include <iostream>
using std::cout;
using std::endl;

#include <fstream>
using std::ifstream;

#include <cstring>
#include <errno.h>
#include "dataParser.h"

#define MAX_IDS 12

char** indexIDs;

void initArray(int lineSize)
{
  indexIDs = new char*[lineSize];
  for(int i = 0; i < lineSize; ++i)
  {
    indexIDs[i] = new char[MAX_IDS];
    for (int j = 0; j < MAX_IDS; ++j)
    {
      indexIDs[i][j] = -1;
    }
  }
}

int getValueID(int index, char c)
{
  char* IDs = indexIDs[index];
  for (int i = 0; i < MAX_IDS; i++)
  {
    if (IDs[i] < 0)
    {
      IDs[i] = c;
      return (int) i;
    }
    if (IDs[i] == c)
    {
      return i;
    }
  }

  return -1;
}

void getTestData(const char* file,
    double *trainingDataSet,
    int lineSize,
    int isImage)
{
    printf("get datas.\n");
  if (!isImage)
    initArray(lineSize);

  std::fstream fin;
  fin.open(file); // open a file
  if (!fin.good())
  {
        std::cerr << "Error: " << strerror(errno);
    return; // exit if file not found
  }

  int line = 0;

  while (!fin.eof())
  {
    int buffSize = lineSize*2 + 5;
    char buf[buffSize];
    fin.getline(buf, buffSize);

    if (!fin.good()) {
        std::cerr << "Error: " << strerror(errno);
        return;
    }

    int n = 0;

    const char* token[lineSize];

    token[0] = strtok(buf, DELIMITER); // first token
    if (token[0]) // zero if line is blank
    {
      for (n = 1; n < lineSize; n++)
      {
        token[n] = strtok(0, DELIMITER); // subsequent tokens
        if (!token[n]) break; // no more tokens
      }
    }

    for (int i=0; i < n; i++)
    {
      if (i)
      {
        trainingDataSet[line*lineSize + i - 1] = isImage? (double)(*token[i])/10.0 : getValueID(i, *token[i]);
      } else {
        trainingDataSet[line*lineSize + lineSize - 1] = isImage? (double)(*token[i])/10.0 : getValueID(i, *token[i]);
      }
    }

    line++;

  }
    printf("done.\n");

  fin.close();

  delete [] indexIDs;
}

