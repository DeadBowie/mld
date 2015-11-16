// DynKnn4.cpp : Defines the entry point for the console application.
//

#include "ANN/ANn.h"
#include <string.h>
#include <stdio.h>
#include "stdafx.h"
#include "DynKnn.h"
#include "mymath.h"
#include "DynKnn4.h"
#include "dynamical.h"
#include <vector>
#include <conio.h>

#define drand48() (rand()/(float)RAND_MAX)


using namespace std;

int GetLabel(int index, ivec labels){ return index < labels.size() ? labels[index] : 0; }


fvec Test(Dynamical *dynamical, vector< vector<fvec> > trajectories, ivec labels)
{
	if (!dynamical || !trajectories.size()) return fvec();
	int dim = trajectories[0][0].size() / 2;
	//(int dim = dynamical->Dim();
	float dT = dynamical->dT;
	fvec sample; sample.resize(dim, 0);
	fvec vTrue; vTrue.resize(dim, 0);
	fvec xMin(dim, FLT_MAX);
	fvec xMax(dim, -FLT_MAX);
	//cout << "Now i'm inside fvec test (DynnKnn4)" << endl;
	// test each trajectory for errors
	int errorCnt = 0;
	float errorOne = 0, errorAll = 0;
	for (int i = 0; i < trajectories.size();i++)
	{
		vector<fvec> t = trajectories[i];
		float errorTraj = 0;
		for (int j = 0; j < t.size();j++)
		{
			for (int d = 0; d< dim;d++)
			{
				sample[d] = t[j][d];
				vTrue[d] = t[j][d + dim];
				if (xMin[d] > sample[d]) xMin[d] = sample[d];
				if (xMax[d] < sample[d]) xMax[d] = sample[d];
			}
			fvec v = dynamical->Test(sample);
			float error = 0;
			for (int d = 0; d < dim;d++) error += (v[d] - vTrue[d])*(v[d] - vTrue[d]);
			errorTraj += error;
			errorCnt++;
		}
		errorOne += errorTraj;
		errorAll += errorTraj / t.size();
	}
	errorOne /= errorCnt;
	errorAll /= trajectories.size();
	fvec res;
	res.push_back(errorOne);
	//cout << "Hello" << endl;
	vector<fvec> endpoints;

	float errorTarget = 0;
	// test each trajectory for target
	fvec pos(dim), end(dim);
	for (int i = 0; i < trajectories.size();i++)
	{
		for (int d = 0; d < dim; d++)
		{
			pos[d] = trajectories[i].front()[d];
			end[d] = trajectories[i].back()[d];
		}
		if (!endpoints.size()) endpoints.push_back(end);
		else
		{
			bool bExists = false;
			for (int j = 0; j < endpoints.size();j++)
			{
				if (endpoints[j] == end)
				{
					bExists = true;
					break;
				}
			}
			if (!bExists) endpoints.push_back(end);
		}
		int steps = 500;
		float eps = FLT_MIN;
		for (int j = 0; j < steps; j++)
		{
			fvec v = dynamical->Test(pos);
			float speed = 0;
			for (int d = 0; d < dim; d++) speed += v[d] * v[d];
			speed = sqrtf(speed);
			if (speed*dT < eps) break;
			pos += v*dT;
		}
		float error = 0;
		for (int d = 0; d < dim; d++)
		{
			error += (pos[d] - end[d])*(pos[d] - end[d]);
		}
		error = sqrtf(error);
		errorTarget += error;
	}
	errorTarget /= trajectories.size();
	res.push_back(errorTarget);
	//cout << "test (DynnKnn4) already over" << endl;
	fvec xDiff = xMax - xMin;
	errorTarget = 0;
	int testCount = 30;
	cout << "Wait a sec"<<endl;
	for (int i = 0; i < testCount;i++)
	{
		for (int d = 0; d < dim; d++)
		{
			pos[d] = ((drand48() * 2 - 0.5)*xDiff[d] + xMin[d]);
		}
		//cout << "Now test (DynnKnn4) already over" << endl;
		int steps = 500;
		float eps = FLT_MIN;
		for (int j = 0; j < steps; j++)
		{
			fvec v = dynamical->Test(pos);
			float speed = 0;
			for (int d = 0; d < dim; d++) speed += v[d] * v[d];
			speed = sqrtf(speed);
			if (speed*dT < eps) break;
			pos += v*dT;
		}
		float minError = FLT_MAX;
		for (int j = 0; j < endpoints.size();j++)
		{
			float error = 0;
			for (int d = 0; d < dim; d++)
			{
				error += (pos[d] - endpoints[j][d])*(pos[d] - endpoints[j][d]);
			}
			error = sqrtf(error);
			if (minError > error) minError = error;
		}
		errorTarget += minError;
		//cout << errorTarget << endl;
	}
	errorTarget /= testCount;
	res.push_back(errorTarget);
	//cout << "test (DynnKnn4) already over" << endl;
	return res;
}



vector< vector < fvec > > GetTrajectories(vector<fvec> samples, vector< ipair > sequences, ivec labels)
{
	int resampleType = 0;// optionsDynamic->resampleCombo->currentIndex();
	int resampleCount = 100;// optionsDynamic->resampleSpin->value();
	int centerType = 0;// optionsDynamic->centerCombo->currentIndex();
	float dT = 10.f;// optionsDynamic->dtSpin->value();
	int zeroEnding = 0;// optionsDynamic->zeroCheck->isChecked();
	bool bColorMap = 0;// optionsDynamic->colorCheck->isChecked();

	// we split the data into trajectories
	vector< vector<fvec> > trajectories;
	if (!sequences.size() || !samples.size()) return trajectories;
	int dim = samples[0].size();
	trajectories.resize(sequences.size());
	for (int i = 0; i < sequences.size(); i++)
	{
		
		int length = sequences[i].second - sequences[i].first + 1;
		trajectories[i].resize(length);
		for (int j = 0; j < length; j++)
		{

			trajectories[i][j].resize(dim * 2);
			// copy data
			for (int d = 0; d < dim; d++) { trajectories[i][j][d] = samples[sequences[i].first + j][d];  }
		}
		
	}

	switch (resampleType)
	{
	case 0: // none
	{
		for (int i = 0; i< sequences.size(); i++)
		{
			int cnt = sequences[i].second - sequences[i].first + 1;
			if (resampleCount > cnt) resampleCount = cnt;
		}
		for (int i = 0; i < trajectories.size(); i++)
		{
			while (trajectories[i].size() > resampleCount) trajectories[i].pop_back();
		}
	}
	break;
	case 1: // uniform
	{
		for (int i = 0; i < trajectories.size(); i++)
		{
			vector<fvec> trajectory = trajectories[i];
			trajectories[i] = interpolate(trajectory, resampleCount);
		}
	}
	break;
	/*case 2: // spline
	{
	for (int i = 0; i < trajectories.size(); i++)
	{
	vector<fvec> trajectory = trajectories[i];
	trajectories[i] = interpolateSpline(trajectory, resampleCount);
	}
	}*/
	break;
	}


	if (centerType)
	{
		map<int, int> counts;
		map<int, fvec> centers;
		vector<int> trajLabels(sequences.size());
		for (int i = 0; i< sequences.size(); i++)
		{
			int index = centerType == 1 ? sequences[i].second : sequences[i].first; // start
			int label = GetLabel(index, labels);
			trajLabels[i] = label;
			if (!centers.count(label))
			{
				fvec center(dim, 0);
				centers[label] = center;
				counts[label] = 0;
			}
			centers[label] += samples[index];
			counts[label]++;
		}
		for (map<int, int>::iterator p = counts.begin(); p != counts.end(); ++p)
		{
			int label = p->first;
			centers[label] /= p->second;
		}
		for (int i = 0; i < trajectories.size(); i++)
		{
			if (centerType == 1)
			{
				fvec difference = centers[trajLabels[i]] - trajectories[i].back();
				for (int j = 0; j < resampleCount; j++) trajectories[i][j] += difference;
			}
			else
			{
				fvec difference = centers[trajLabels[i]] - trajectories[i][0];
				for (int j = 0; j < resampleCount; j++) trajectories[i][j] += difference;
			}
		}
	}

	float maxV = -FLT_MAX;
	// we compute the velocity
	for (int i = 0; i < trajectories.size(); i++)
	{
		for (int j = 0; j < resampleCount - 1; j++)
		{
			for (int d = 0; d< dim; d++)
			{
				float velocity = (trajectories[i][j + 1][d] - trajectories[i][j][d]) / dT;
				trajectories[i][j][dim + d] = velocity;
				if (velocity > maxV) maxV = velocity;

			}

		}
		if (!zeroEnding)
		{
			for (int d = 0; d< dim; d++)
			{
				trajectories[i][resampleCount - 1][dim + d] = trajectories[i][resampleCount - 2][dim + d];
			}
		}
	}

	// we normalize the velocities as the variance of the data
	fvec mean, sigma;
	mean.resize(dim, 0);
	int cnt = 0;
	sigma.resize(dim, 0);
	for (int i = 0; i < trajectories.size(); i++)
	{
		for (int j = 0; j < resampleCount; j++)
		{
			mean += trajectories[i][j];
			cnt++;
		}
	}
	mean /= cnt;
	for (int i = 0; i < trajectories.size(); i++)
	{
		for (int j = 0; j < resampleCount; j++)
		{
			fvec diff = (mean - trajectories[i][j]);
			for (int d = 0; d< dim; d++) sigma[d] += diff[d] * diff[d];
		}
	}
	sigma /= cnt;

	for (int i = 0; i < trajectories.size(); i++)
	{
		for (int j = 0; j < resampleCount; j++)
		{
			for (int d = 0; d< dim; d++)
			{
				trajectories[i][j][dim + d] /= maxV;
				//trajectories[i][j][dim + d] /= sqrt(sigma[d]);
			}
		}
	}
	return trajectories;
}


fvec Train(Dynamical *dynamical, vector<fvec> samples, vector< ipair > sequences, ivec labels)
{
	if(!dynamical) return fvec();
	
	if (!samples.size() || !sequences.size()) return fvec();
	int dim = samples[0].size();
	int resampleType = 0;// optionsDynamic->resampleCombo->currentIndex();
	int count = 100;// optionsDynamic->resampleSpin->value();
	int centerType = 0;// optionsDynamic->centerCombo->currentIndex();
	float dT = 10.f;// optionsDynamic->dtSpin->value();
	int zeroEnding = 0;// optionsDynamic->zeroCheck->isChecked();
	
	
	ivec trajLabels(sequences.size());
	for (int i = 0; i < sequences.size(); i++)
	{
		trajLabels[i] = labels[sequences[i].first];
	}

	//float dT = 10.f; // time span between each data frame
	
	dynamical->dT = dT;
	//dT = 10.f;
	vector< vector<fvec> > trajectories = GetTrajectories(samples, sequences, labels);
	interpolate(trajectories[0], count);
	
	
	dynamical->Train(trajectories, labels);
	//cout << "I'm here" << endl;
	return Test(dynamical, trajectories, labels);

}







void Dynamize(vector<fvec> &samples, vector< ipair > &sequences, ivec &labels)
{
	//if (!canvas || !canvas->data->GetCount() || !canvas->data->GetSequences().size()) return;
	//drawTimer->Stop();
	//drawTimer->Clear();
	//QMutexLocker lock(&mutex);
	//DEL(clusterer);
	//DEL(regressor);
	//delete dynamical;
	//DEL(classifier);
	//DEL(maximizer);
	//DEL(reinforcement);
	//DEL(projector);
	//lastTrainingInfo = "";
	//if (!optionsDynamic->algoList->count()) return;
	//int tab = optionsDynamic->algoList->currentIndex();
	//if (tab >= dynamicals.size() || !dynamicals[tab]) return;
	//dynamical = dynamicals[tab]->GetDynamical();
	//tabUsedForTraining = tab;
	//dynamical = dynamicals[tab]->GetDynamical();
	cout << "Dynamize"<<endl;
	DynamicalKNN *dynamical = new DynamicalKNN();
	((DynamicalKNN *)dynamical)->SetParams(2, 1, 5);//(k, metricType, metricP);
	Train(dynamical,samples,sequences,labels);
	//dynamicals[tab]->Draw(canvas, dynamical);

	//int w = canvas->width(), h = canvas->height();
	for (int xdim = 0; xdim < 20; xdim++){
		for (int ydim = 0; ydim < 20; ydim++){
	ivec labels1 = {1};
	vector<fvec> samples1 = { { float(xdim)/2, float(ydim)/2 }, { float(xdim + 1)/2, float(ydim+1)/2 } };
	vector< ipair > sequences1 = { { 0, 1 } };

	// we draw the current trajectories
	
	vector< vector<fvec> > trajectories = GetTrajectories(samples1, sequences1, labels1);
		//GetTrajectories(samples, sequences, labels); 
	/*for (int i = 0; i < trajectories.size(); i++) {
		cout << "one" << endl;
		for (int j = 0; j < trajectories[i].size(); j++)
		{
			cout << "q"<<endl;
			for (int k = 0; k < trajectories[i][j].size(); k++)  cout << trajectories[i][j][k] << "   ";
		}
	}*/
	//_getch();
	vector< vector<fvec> > testTrajectories(400);
	int steps = 3;
	if (trajectories.size())
	{
		testTrajectories.resize(trajectories.size());
		int dim = trajectories[0][0].size() / 2;
		for (int i = 0; i < trajectories.size(); i++)
		{
			fvec start(dim, 0);
			for (int d = 0; d < dim; d++) start[d] = trajectories[i][0][d];
			vector<fvec> result = dynamical->Test(start, steps);
			testTrajectories[i] = result;


		}

		/*
		for (int i = 0; i < testTrajectories.size(); i++) {
		cout << "one" << endl;
		for (int j = 0; j < testTrajectories[i].size(); j++)
		{
		cout << "q" << endl;
		for (int k = 0; k < testTrajectories[i][j].size(); k++)  cout << testTrajectories[i][j][k] << "   ";
		}
		_getch();
		}
		_getch();*/
		//canvas->maps.model = QPixmap(w, h);
		//QBitmap bitmap(w, h);
		//bitmap.clear();
		//canvas->maps.model.setMask(bitmap);
		//canvas->maps.model.fill(Qt::transparent);


		for (int i = 0; i < testTrajectories.size(); i++)
		{
			vector<fvec> &result = testTrajectories[i];
			fvec oldPt = result[0];
			int count = result.size();
			for (int j = 1; j < count - 1; j++)
			{
				fvec pt = result[j + 1];


				cout << "[" << oldPt[0] << "," << oldPt[1] << "]->[" << pt[0] << "," << pt[1] << "]" << endl;
				oldPt = pt;
			}
			}
		}
	}
			/*
			painter.setBrush(Qt::NoBrush);
			painter.setPen(Qt::green);
			painter.drawEllipse(canvas->toCanvasCoords(result[0]), 5, 5);
			painter.setPen(Qt::red);
			painter.drawEllipse(canvas->toCanvasCoords(result[count-1]), 5, 5);}*/


			//}
			//else
			//{
			//pair<fvec, fvec> bounds = canvas->data->GetBounds();
			//Expose::DrawTrajectories(canvas->maps.model, testTrajectories, vector<QColor>(), canvas->canvasType - 1, 1, bounds);
			//}
		
	}

	// the first index is "none", so we subtract 1
	/*int avoidIndex = optionsDynamic->obstacleCombo->currentIndex() - 1;
	if (avoidIndex >= 0 && avoidIndex < avoiders.size() && avoiders[avoidIndex])
	{
		DEL(dynamical->avoid);
		dynamical->avoid = avoiders[avoidIndex]->GetObstacleAvoidance();
	}
	UpdateInfo();
	if (dynamicals[tab]->UsesDrawTimer())
	{
		drawTimer->bColorMap = bColorMap;
		drawTimer->start(QThread::NormalPriority);
	}*/
}



void AddSample(fvec sample, vector<fvec>  &samples /*, dsmFlags flag*/)
{
	if (!sample.size()) return;
	int size = sample.size();

	samples.insert(samples.end(), sample);
	
}
void AddSequence(ipair newSequence, vector< ipair > &sequences)
{
	//if (newSequence.first >= samples.size() || newSequence.second >= samples.size()) return;
	
	sequences.push_back(newSequence);
	// sort sequences by starting value
	sort(sequences.begin(), sequences.end());
	
}


int main()
{
	DynamicalKNN *dyn = new DynamicalKNN;
	//std::vector< fvec > samples;
	ipair trajectory = {-1,-1};
	ivec labels;
	vector<fvec> samples;
	vector<int> flags;
	fvec lol;
	vector< ipair > sequences;
	/*for (int i = 0; i < 5; i++){
		float k = i;
		 lol = { k, k };
		AddSample(lol, samples);
		labels.insert(labels.end(), 1);
		flags.insert(flags.end, 1); //2=traj
	}
	*/
	int a; int l;
	cout << "kol-vo trajectoriy "; cin >> a;
	for (int j = 0; j < a; j++){
		cout << "kol-vo tochek " << j+1 << " traj ";
		cin >> l;
		for (int i = 0; i < l; i++){
			if (trajectory.first == -1) // we're starting a trajectory
				trajectory.first = samples.size();//canvas->data->GetCount();
			// we don't want to draw too often
			//if(drawTime.elapsed() < 50/speed) return; // msec elapsed since last drawing
			float x, y;
			cout << i+1<<":" << endl << "x="; cin >> x; cout << "y=";cin >> y;
			lol = { x, y };
			AddSample(lol, samples);
			flags.push_back(2);
			labels.push_back(j);
			trajectory.second = samples.size() - 1;
		}

		if (trajectory.first != -1)
		{
			// the last point is a duplicate, we take it out
			AddSequence(trajectory, sequences);
			trajectory.first = -1;
			//for (int i = trajectory.first; i <= trajectory.second; i++) 
			//labels[i] = 2;
		}
	}
	/*
	for (int i = 0; i < 4; i++){
		if (trajectory.first == -1) // we're starting a trajectory
		{
			trajectory.first = samples.size();//canvas->data->GetCount();
		}
		// we don't want to draw too often
		//if(drawTime.elapsed() < 50/speed) return; // msec elapsed since last drawing
		float k = i;
		lol = { 7-k, k };
		

		AddSample(lol, samples);
		flags.push_back(2);
		labels.push_back(2);
		trajectory.second = samples.size() - 1;
	}

	if (trajectory.first != -1)
	{
		// the last point is a duplicate, we take it out
		AddSequence(trajectory, sequences);
		trajectory.first = -1;
		
	}
	*/
	//(*dyn).dynamical;
	//DynamicalKNN *knn = new DynamicalKNN();
	//Dynamical *dyn = new Dynamical;
	//u32 k = 2;
	//knn->SetParams(k, 1, 5);
	//dyn = knn;


	// we build the trajectory(by hand)

/*	int count = 10;
	vector<fvec> trajectory(30);
	fvec position = { 1, 3 };//sample;
	for (int i = 0; i < count; i++)
	{
		trajectory.push_back(position);
		fvec velocity = dyn->Test(position);
		
		position += velocity*dyn->dT;
		//if (velocity == 0) break;
	}
	*/
	

	
	Dynamize(samples,  sequences, labels);
	//Train(dyn, samples, sequences);
	//Draw(knn);

	system("PAUSE");
	return 0;
	
}



/*
void Train(DynamicalKNN *dynamical)
{
	if (!dynamical) return;
	//std::vector<fvec> samples = Canvas::data.GetSamples();
	std::vector<fvec> samples = { { 1,2 }, { 1,3 }, { 1,4 } };
	std::vector<ipair> sequences = { { 1, 2 }, { 1, 3 }, { 1, 4 } };
	//ivec labels = Canvas::data.GetLabels();
	ivec labels = { 1, 1, 1 };
	//if (!samples.size() || !sequences.size()) return;
	int dim = samples[0].size();
	int count = 5;//dynOptions->resampleSpin->value();
	int resampleType = 0;//dynOptions->resampleCombo->currentIndex();
	int centerType = 0;// dynOptions->centerCombo->currentIndex();
	bool zeroEnding = 0;// dynOptions->zeroCheck->isChecked();

	// we split the data into trajectories
	std::vector< std::vector<fvec> > trajectories;
	ivec trajLabels;
	trajectories.resize(sequences.size());
	trajLabels.resize(sequences.size());
	for (int i = 0; i<sequences.size(); i++)
	{
		int length = sequences[i].second - sequences[i].first + 1;
		trajLabels[i] = { 1 }; //Canvas::data.GetLabel(sequences[i].first);
		trajectories[i].resize(length);
		for (int j = 0; j< length; j++)
		{
			trajectories[i][j].resize(dim * 2);
			// copy data
			for (int d = 0; d< dim; d++) trajectories[i][j][d] = samples[sequences[i].first + j][d];
		}
	}

	switch (resampleType)
	{
	case 0: // none
	{
		for (int i = 0; i<sequences.size(); i++)
		{
			int cnt = sequences[i].second - sequences[i].first + 1;
			if (count > cnt) count = cnt;
		}
		for (int i = 0; i<trajectories.size(); i++)
		{
			while (trajectories[i].size() > count) trajectories[i].pop_back();
		}
	}
	break;
	case 1: // uniform
	{
		for (int i = 0; i<trajectories.size(); i++)
		{
			std::vector<fvec> trajectory = trajectories[i];
			trajectories[i] = interpolate(trajectory, count);
		}
	}
	break;
	}


	if (centerType)
	{
		std::map<int, int> counts;
		std::map<int, fvec> centers;
		for (int i = 0; i<sequences.size(); i++)
		{
			int index = centerType ? sequences[i].second : sequences[i].first; // start
			int label = 1;//Canvas::data.GetLabel(index);
			if (!centers.count(label))
			{
				fvec center;
				center.resize(2, 0);
				centers[label] = center;
				counts[label] = 0;
			}
			centers[label] += samples[index];
			counts[label]++;
		}
		for (map<int, int>::iterator p = counts.begin(); p != counts.end(); ++p)
		{
			int label = p->first;
			centers[label] /= p->second;
		}
		for (int i = 0; i<trajectories.size(); i++)
		{
			fvec difference = centers[trajLabels[i]] - trajectories[i][count - 1];
			for (int j = 0; j< count; j++) trajectories[i][j] += difference;
		}
	}

	float dT = 10.f; // time span between each data frame
	dynamical->dT = dT;

	float maxV = -FLT_MAX;
	// we compute the velocity
	for (int i = 0; i<trajectories.size(); i++)
	{
		for (int j = 0; j< count - 1; j++)
		{
			for (int d = 0; d< dim; d++)
			{
				float velocity = (trajectories[i][j + 1][d] - trajectories[i][j][d]) * dT;
				trajectories[i][j][dim + d] = velocity;
				if (velocity > maxV) maxV = velocity;
			}
		}
		if (!zeroEnding)
		{
			for (int d = 0; d< dim; d++)
			{
				trajectories[i][count - 1][dim + d] = trajectories[i][count - 2][dim + d];
			}
		}
	}

	for (int i = 0; i<trajectories.size(); i++)
	{
		for (int j = 0; j< count; j++)
		{
			for (int d = 0; d< dim; d++)
			{
				trajectories[i][j][dim + d] /= maxV;
			}
		}
	}

	dynamical->Traind(trajectories, labels);
}
*/
/*
void GetDynamical(int tab)
{
	/*if(tab == 0) // gmr
	{
	DynamicalGMR *gmr = new DynamicalGMR();
	int clusters = dynOptions->gmmCount->value();
	int covType = dynOptions->gmmCovarianceCombo->currentIndex();
	int initType = dynOptions->gmmInitCombo->currentIndex();
	gmr->SetParams(clusters, covType, initType);
	dynamical = gmr;
	}
	else if(tab == 1) // lwpr
	{
	float gen = dynOptions->lwprGenSpin->value();
	float delta = dynOptions->lwprInitialDSpin->value();
	float alpha = dynOptions->lwprAlphaSpin->value();

	DynamicalLWPR *lwpr = new DynamicalLWPR();
	lwpr->SetParams(delta, alpha, gen);
	dynamical = lwpr;
	}
	else if(tab == 2) // kernel methods
	{
	int kernelMethod = dynOptions->svmTypeCombo->currentIndex();
	float svmC = dynOptions->svmCSpin->value();
	int kernelType = dynOptions->kernelTypeCombo->currentIndex();
	float kernelGamma = dynOptions->kernelWidthSpin->value();
	float kernelDegree = dynOptions->kernelDegSpin->value();
	float svmP = dynOptions->svmPSpin->value();

	if(kernelMethod == 2) // sogp
	{
	DynamicalGPR *gpr = new DynamicalGPR();
	int capacity = svmC;
	double kernelNoise = svmP;
	gpr->SetParams(kernelGamma, kernelNoise, capacity, kernelType, kernelDegree);
	dynamical = gpr;
	}
	else
	{
	DynamicalSVR *svm = new DynamicalSVR();
	switch(kernelMethod)
	{
	case 0:
	svm->param.svm_type = EPSILON_SVR;
	break;
	case 1:
	svm->param.svm_type = NU_SVR;
	break;
	}
	switch(kernelType)
	{
	case 0:
	svm->param.kernel_type = LINEAR;
	break;
	case 1:
	svm->param.kernel_type = POLY;
	break;
	case 2:
	svm->param.kernel_type = RBF;
	break;
	}
	svm->param.C = svmC;
	svm->param.nu = svmP;
	svm->param.p = svmP;
	svm->param.gamma = 1 / kernelGamma;
	svm->param.degree = kernelDegree;
	dynamical = svm;
	}
	}
	else if(tab == 3) // mlp
	{
	float alpha = dynOptions->mlpAlphaSpin->value();
	float beta = dynOptions->mlpBetaSpin->value();
	int layers = dynOptions->mlpLayerSpin->value();
	int neurons = dynOptions->mlpNeuronSpin->value();
	int activation = dynOptions->mlpFunctionCombo->currentIndex()+1; // 1: sigmoid, 2: gaussian

	DynamicalMLP *mlp = new DynamicalMLP();
	mlp->SetParams(activation, neurons, layers, alpha, beta);
	dynamical = mlp;
	}
	else if(tab == 4) // aknn
	{
	//knnKspin->setMinimum(1);knnKspin->setMaximum(99);
	//int k = dynOptions->knnKspin->value();
	int k = 2;
	// int metricType = dynOptions->knnNormCombo->currentIndex();
	int metricType = 0;
	//knnNormSpin->setMinimum(1); knnNormSpin->setMaximum(20);
	// int metricP = dynOptions->knnNormSpin->value();
	int metricP = 2;
	DynamicalKNN *knn = new DynamicalKNN();
	knn->SetParams(k, metricType, metricP);
	dynamical = knn;
	//}
}*/



