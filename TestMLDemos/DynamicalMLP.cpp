/*********************************************************************
MLDemos: A User-Friendly visualization toolkit for machine learning
Copyright (C) 2010  Basilio Noris
Contact: mldemos@b4silio.com

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Library General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free
Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*********************************************************************/
#include "DynamicalMLP.h"

using namespace std;

DynamicalMLP::DynamicalMLP()
	: functionType(1), neuronCount(2), mlp(0), alpha(0), beta(0), dT(0.02f)
{
}

DynamicalMLP::~DynamicalMLP()
{
	delete mlp;
}

void DynamicalMLP::Train(std::vector< std::vector<fvec> > trajectories, ivec labels)
{
	if (!trajectories.size()) return;
	int count = trajectories[0].size();
	if (!count) return;
	dim = trajectories[0][0].size() / 2;
	// we forget about time and just push in everything
	vector<fvec> samples;
	for (int i = 0; i < trajectories.size(); i++)
	{
		for (int j = 0; j < trajectories[i].size(); j++)
		{
			samples.push_back(trajectories[i][j]);
		}
	}
	u32 sampleCnt = samples.size();
	if (!sampleCnt) return;
	delete mlp;

	CvMat *layers;
	//	if(neuronCount == 3) neuronCount = 2; // don't ask me why but 3 neurons mess up everything...

	if (!layerCount || neuronCount < 2)
	{
		layers = cvCreateMat(2, 1, CV_32SC1);
		cvSet1D(layers, 0, cvScalar(dim));
		cvSet1D(layers, 1, cvScalar(dim));
	}
	else
	{
		layers = cvCreateMat(2 + layerCount, 1, CV_32SC1);
		cvSet1D(layers, 0, cvScalar(dim));
		cvSet1D(layers, layerCount + 1, cvScalar(dim));
		for (int i = 0; i < layerCount; i++) cvSet1D(layers, i + 1, cvScalar(neuronCount));
	}

	u32 *perm = randPerm(sampleCnt);

	CvMat *trainSamples = cvCreateMat(sampleCnt, dim, CV_32FC1);
	CvMat *trainOutputs = cvCreateMat(sampleCnt, dim, CV_32FC1);
	CvMat *sampleWeights = cvCreateMat(samples.size(), 1, CV_32FC1);
	for (int i = 0; i < sampleCnt; i++)
	{
		for (int j = 0; j < dim; j++) cvSetReal2D(trainSamples, i, j, samples[perm[i]][j]);
		for (int j = 0; j < dim; j++) cvSetReal2D(trainOutputs, i, j, samples[perm[i]][dim + j]);
		cvSet1D(sampleWeights, i, cvScalar(1));
	}

	delete[] perm;

	int activationFunction = functionType == 2 ? CvANN_MLP::GAUSSIAN : functionType ? CvANN_MLP::SIGMOID_SYM : CvANN_MLP::IDENTITY;


	mlp = new CvANN_MLP();
	mlp->create(layers, activationFunction, alpha, beta);

	CvANN_MLP_TrainParams params;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1000, 0.001);
	mlp->train(trainSamples, trainOutputs, sampleWeights, 0, params);
	cvReleaseMat(&trainSamples);
	cvReleaseMat(&trainOutputs);
	cvReleaseMat(&sampleWeights);
	cvReleaseMat(&layers);
}

std::vector<fvec> DynamicalMLP::Test(const fvec &sample, const int count)
{
	fvec start = sample;
	dim = sample.size();
	std::vector<fvec> res(count);
	for (int i = 0; i < count; i++) res[i].resize(dim, 0);
	if (!mlp) return res;

	float *_input = new float[dim];
	CvMat input = cvMat(1, dim, CV_32FC1, _input);
	float *_output = new float[dim];
	CvMat output = cvMat(1, dim, CV_32FC1, _output);
	fvec velocity; velocity.resize(dim, 0);
	for (int i = 0; i < count; i++)
	{
		res[i] = start;
		for (int j = 0; j < start.size(); j++) {
		start[j] += velocity[j]*dT;
		}
		for (int d = 0; d < dim; d++) _input[d] = start[d];
		mlp->predict(&input, &output);
		for (int d = 0; d < dim; d++) velocity[d] = _output[d];
	}
	delete[] _input;
	delete[] _output;
	return res;
}

fvec DynamicalMLP::Test(const fvec &sample)
{
	int dim = sample.size();
	fvec res(2);
	if (!mlp) return res;
	float *_input = new float[dim];
	for (int d = 0; d < dim; d++) _input[d] = sample[d];
	CvMat input = cvMat(1, dim, CV_32FC1, _input);
	float *_output = new float[dim];
	CvMat output = cvMat(1, dim, CV_32FC1, _output);
	mlp->predict(&input, &output);
	for (int d = 0; d < dim; d++) res[d] = _output[d];
	delete[] _input;
	delete[] _output;
	return res;
}

void DynamicalMLP::SetParams(u32 functionType, u32 neuronCount, u32 layerCount, f32 alpha, f32 beta)
{
	this->functionType = functionType;
	this->neuronCount = neuronCount;
	this->layerCount = layerCount;
	this->alpha = alpha;
	this->beta = beta;
}


const char *DynamicalMLP::GetInfoString()
{
	stringstream out;
	out << "Multi-Layer Perceptron\n";
	out << "Layers: " << layerCount << endl;
	out << "Neurons: " << neuronCount << endl;
	out << "Activation Function: " << endl;
	switch (functionType)
	{
	case 0:
		out << " identity\n";
		break;
	case 1:
		out << "sigmoid (alpha: "<< alpha << " beta: "<< beta << ")\n\tbeta*(1-exp(-alpha*x)) / (1 + exp(-alpha*x))\n";
		break;
	case 2:
		out << " gaussian (" << alpha << " beta: " << beta << ")\n\tbeta*exp(-alpha*x*x)\n";
		break;
	}
	auto result = out.str();
	cout << result << endl;
	char *text = new char[1024];
	
	return text;
}
