#include <math.h>

typedef float Logfloat;
typedef double LogDouble;

#define LZERO (-1.0E10)
#define LSMALL (-0.5E10) ////if log(x) < LSMALL regnore it

static double minLogExp = -log(-LZERO);

LogDouble Logf(double x);
LogDouble LAdd(LogDouble x, LogDouble y);