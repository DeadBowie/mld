#pragma once

/*********************************************************************
MLDemos: A User-Friendly visualization toolkit for machine learning
Copyright (C) 2010  Basilio Noris
Contact: mldemos@b4silio.com

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License,
version 3 as published by the Free Software Foundation.

This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free
Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*********************************************************************/
#ifndef _DYNAMICAL_MLP_H_
#define _DYNAMICAL_MLP_H_

#include <vector>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

typedef float f32;
typedef std::vector<float> fvec;
typedef std::vector<int> ivec;
typedef int s32;
typedef unsigned int u32;

class DynamicalMLP
{
private:
	static u32 *randPerm(u32 length, s32 seed = -1)
	{
		u32 *perm = new u32[length];
		memset(perm, 0, length*sizeof(u32));

		u32 *usable = new u32[length];
		for (int i = 0; i < length; i++) usable[i] = i;

		if (seed == -1) srand((u32)cvGetTickCount());
		else srand((u32)seed);

		u32 uLength = length;
		for (register u32 i = 0; i<length; i++)
		{
			register u32 r = ((rand() << 7) + rand()) % uLength;

			perm[i] = usable[r];
			uLength--;
			usable[r] = usable[uLength];
			usable[uLength] = 0;
		}
		delete[] usable;

		return perm;
	}


	u32 dim;
	float dT;
	u32 functionType; // 1: sigmoid, 2: gaussian
	u32 neuronCount;
	u32 layerCount;
	float alpha, beta;
	CvANN_MLP *mlp;
public:
	DynamicalMLP();
	~DynamicalMLP();
	void Train(std::vector< std::vector<fvec> > trajectories, ivec labels);
	std::vector<fvec> Test(const fvec &sample, const int count);
	fvec Test(const fvec &sample);
	const char *GetInfoString();

	void SetParams(u32 functionType, u32 neuronCount, u32 layerCount, f32 alpha, f32 beta);
};

#endif // _DYNAMICAL_MLP_H_
