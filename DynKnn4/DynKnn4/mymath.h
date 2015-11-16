
#ifndef _MY_MATHS_H
#define _MY_MATHS_H


#include <vector>

typedef std::vector<float> fvec;
int min(int a, int b);
void operator += (fvec &a, const fvec b);
void operator -= (fvec &a, const fvec b);
void operator += (fvec &a, const float b);
void operator -= (fvec &a, const float b);
void operator *= (fvec &a, const float b);
void operator /= (fvec &a, const float b);
fvec operator + (const fvec a, const fvec b);
fvec operator - (const fvec a, const fvec b);
fvec operator + (const fvec a, const float b);
fvec operator - (const fvec a, const float b);
fvec operator * (const fvec a, const float b);
fvec operator / (const fvec a, const float b);
float operator * (const fvec a, const fvec b);

std::vector<fvec> interpolate(std::vector<fvec> a, int count);

#endif // _MY_MATHS_H
