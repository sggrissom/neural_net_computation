#include <iostream>
using std::cout;
using std::endl;

#include <fstream>
using std::ifstream;

#include <cstring>
#include "dataParser.h"

char** indexIDs;

void initArray()
{
  indexIDs = new char*[MAX_TOKENS_PER_LINE];
  for(int i = 0; i < MAX_TOKENS_PER_LINE; ++i)
  {
    indexIDs[i] = new char[MAX_TOKEN_TYPES];
    for (int j = 0; j < MAX_TOKEN_TYPES; ++j)
    {
      indexIDs[i][j] = -1;
    }
  }
}

int getValueID(int index, char c)
{
  char* IDs = indexIDs[index];
  for (int i = 0; i < 10; i++)
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
    double *trainingDataSet)
{
  initArray();

  std::fstream fin;
  fin.open(file); // open a file
  if (!fin.good())
  {
    return; // exit if file not found
  }

  int line = 0;

  while (!fin.eof())
  {
    char buf[MAX_CHARS_PER_LINE];
    fin.getline(buf, MAX_CHARS_PER_LINE);

    int n = 0;

    const char* token[MAX_TOKENS_PER_LINE] = {};

    token[0] = strtok(buf, DELIMITER); // first token
    if (token[0]) // zero if line is blank
    {
      for (n = 1; n < MAX_TOKENS_PER_LINE; n++)
      {
        token[n] = strtok(0, DELIMITER); // subsequent tokens
        if (!token[n]) break; // no more tokens
      }
    }

    for (int i=0; i < n; i++)
    {
      if (i)
      {
        trainingDataSet[line*MAX_TOKENS_PER_LINE + i - 1] = getValueID(i, *token[i]);
      } else {
        trainingDataSet[line*MAX_TOKENS_PER_LINE + MAX_TOKENS_PER_LINE - 1] = getValueID(i, *token[i]);
      }
    }

    line++;
  }

  fin.close();

  delete [] indexIDs;
}

