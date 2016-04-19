/*
 * Copyright (c) 2009-2010, Sergey Karakovskiy and Julian Togelius
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Mario AI nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package ch.idsia.scenarios;

import java.io.Serializable;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.FileOutputStream;
import java.util.Random;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import ch.idsia.agents.Agent;
import ch.idsia.agents.learning.SupervisedLearningAgent;
import edu.stanford.cs229.agents.MarioRLAgent;
import ch.idsia.agents.learning.WekaRecordingAgent;
import ch.idsia.agents.controllers.ForwardAgent;
import ch.idsia.benchmark.mario.environments.Environment;
import ch.idsia.benchmark.tasks.BasicTask;
import ch.idsia.tools.MarioAIOptions;


import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Attribute;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.FastVector;
import ch.idsia.agents.Agent;
import ch.idsia.benchmark.mario.engine.sprites.Mario;
import ch.idsia.benchmark.mario.environments.Environment;
import ch.idsia.agents.controllers.BasicMarioAIAgent;
import ch.idsia.evolution.Evolvable;
import ch.idsia.evolution.MLP;
import java.util.ArrayList;
import java.util.List;

import ch.idsia.agents.LearningAgent;
import ch.idsia.benchmark.tasks.LearningTask;
import ch.idsia.tools.EvaluationInfo;
import ch.idsia.tools.MarioAIOptions;
import edu.stanford.cs229.agents.ActionQtable;
import edu.stanford.cs229.agents.MarioAction;
import edu.stanford.cs229.agents.MarioState;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Random;
import weka.core.Instance;
import weka.core.*;
import weka.core.Instances;

import weka.classifiers.*;
import weka.filters.unsupervised.instance.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.functions.supportVector.PolyKernel;

/**
 * Created by IntelliJ IDEA. User: Sergey Karakovskiy, sergey at idsia dot ch Date: Mar 17, 2010 Time: 8:28:00 AM
 * Package: ch.idsia.scenarios
 */
public final class Main
{
	public static boolean recordingMode = false;

	public static boolean use10foldcross = true;
	public static void TrainTestSplit(Instances data, Classifier scheme, String name) throws Exception{
		StratifiedRemoveFolds filter = new StratifiedRemoveFolds();
		String[] options = new String[6];
		options[0] = "-N";                  // Indicate we want to set the number of folds
		options[1] = Integer.toString(5);   // Split the data into 5 random folds
		options[2] = "-F";                  // Indicate we want to select a specific fold
		options[3] = Integer.toString(1);   // Select the first fold
		options[4] = "-S";                  // Indicate we want to set the random seed
		options[5] = Integer.toString(1);   // Set the random seed to 1

		filter.setOptions(options);
		filter.setInputFormat(data);
		filter.setInvertSelection(false);
		Instances test = Filter.useFilter(data, filter);
		filter.setInvertSelection(true);
		Instances train = Filter.useFilter(data, filter);
		
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(scheme,test);
		System.out.println(eval.toSummaryString("\n" + name + " (Train/Test) Results\n======\n",true));
		System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
	}
	
	public static RandomForest RandomForestTree(Instances data) throws Exception{
		RandomForest rf = new RandomForest();
		rf.buildClassifier(data);
		if (use10foldcross) {
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(rf, data, 10, new Random(1));
			System.out.println(eval.toSummaryString("\nRandom Forest Results\n======\n",true));
			System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
		}
		else {
			TrainTestSplit(data,rf,"Random Forest");
		}
		return rf;
	}
	
	public static void BogoTestCase() throws Exception{
		 // Declare two numeric attributes
		
		
		 Attribute Attribute1 = new Attribute("firstNumeric");
		 Attribute Attribute2 = new Attribute("secondNumeric");
		 
		 // Declare a nominal attribute along with its values
		 FastVector fvNominalVal = new FastVector(3);
		 fvNominalVal.addElement("blue");
		 fvNominalVal.addElement("gray");
		 fvNominalVal.addElement("black");
		 Attribute Attribute3 = new Attribute("aNominal", fvNominalVal);
		 
		 // Declare the class attribute along with its values
		 FastVector fvClassVal = new FastVector(2);
		 fvClassVal.addElement("positive");
		 fvClassVal.addElement("negative");
		 Attribute ClassAttribute = new Attribute("theClass", fvClassVal);
		 
		 // Declare the feature vector
		 FastVector fvWekaAttributes = new FastVector(4);
		 fvWekaAttributes.addElement(Attribute1);
		 fvWekaAttributes.addElement(Attribute2);
		 fvWekaAttributes.addElement(Attribute3);
		 fvWekaAttributes.addElement(ClassAttribute);
		 
		// Create an empty training set
		 Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, 10);
		 // Set class index
		 isTrainingSet.setClassIndex(3);
		 
		// Create the instance
		 Instance iExample = new SparseInstance(4);
		 iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), 1.0);
		 iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), 0.5);
		 iExample.setValue((Attribute)fvWekaAttributes.elementAt(2), "gray");
		 iExample.setValue((Attribute)fvWekaAttributes.elementAt(3), "positive");
		 
		 // add the instance
		 isTrainingSet.add(iExample);
		 
		// Create a naïve bayes classifier
		 Classifier cModel = (Classifier)new NaiveBayes();
		 cModel.buildClassifier(isTrainingSet);
		 // Specify that the instance belong to the training set
		 // in order to inherit from the set description
		 Instance iUse = new SparseInstance(4);
		 iUse.setValue((Attribute)fvWekaAttributes.elementAt(0), 1.0);
		 iUse.setValue((Attribute)fvWekaAttributes.elementAt(1), 0.5);
		 iUse.setValue((Attribute)fvWekaAttributes.elementAt(2), "gray");
		 
		 iUse.setDataset(isTrainingSet);
		 
		 // Get the likelihood of each classes
		 // fDistribution[0] is the probability of being positive
		 // fDistribution[1] is the probability of being negative
		 double[] fDistribution = cModel.distributionForInstance(iUse);
		 
		 System.out.println(fDistribution[0]);
		 System.out.println(fDistribution[1]);
		 
	}

	public static void SerializeModel() throws Exception, IOException{
		 // create J48
		 Classifier cls = new J48();
		 
		 // train
		 Instances inst = new Instances(
		                    new BufferedReader(
		                      new FileReader("E:\\Development\\SchoolWork Spring 2016\\Learning and Advanced Game AI\\SupervisedLearningForMario\\cs229mario\\Data\\trainingData.arff")));
		 inst.setClassIndex(inst.numAttributes() - 1);
		 cls.buildClassifier(inst);
		 
		 // serialize model
		 ObjectOutputStream oos = new ObjectOutputStream(
		                            new FileOutputStream("E:\\Development\\SchoolWork Spring 2016\\Learning and Advanced Game AI\\SupervisedLearningForMario\\cs229mario\\Data\\j48.model"));
		 oos.writeObject(cls);
		 oos.flush();
		 oos.close();
		
	}
	
	public static Classifier DeserializeModel() throws Exception, IOException{
		// deserialize model
		 ObjectInputStream ois = new ObjectInputStream(
		                           new FileInputStream("E:\\Development\\SchoolWork Spring 2016\\Learning and Advanced Game AI\\SupervisedLearningForMario\\cs229mario\\Data\\j48.model"));
		 Classifier cls = (Classifier) ois.readObject();
		 ois.close();
		 return cls;
	}
	
	public static void main(String[] args) throws Exception
	{
		/*
		BogoTestCase();
		*/
	//   00     final String argsString = "-vis on";
	    final MarioAIOptions marioAIOptions = new MarioAIOptions(args);
	//        final Environment environment = new MarioEnvironment();
	        SupervisedLearningAgent agent = null;
	//        final Agent agent = marioAIOptions.getAgent();
	//        final Agent a = AgentsPool.loadAgent("ch.idsia.controllers.agents.controllers.ForwardJumpingAgent");

	        MarioRLAgent agent2 = null;
	        RandomForest rf = null;
	        //Establish the ai
	        if(!recordingMode){
	    		SerializeModel();
		        BufferedReader breader = null;
				breader = new BufferedReader(new FileReader("E:\\Development\\SchoolWork Spring 2016\\Learning and Advanced Game AI\\SupervisedLearningForMario\\cs229mario\\Data\\trainingData.arff"));
				
				Instances data = new Instances(breader);
				agent = new SupervisedLearningAgent(data);
				data.setClassIndex(data.numAttributes() -1);
				breader.close();
				
				
		        agent.SetClassifier(DeserializeModel());
	        } else if(recordingMode){
		        agent2 = new MarioRLAgent();
	        }
	        
	        //agent.SetRandomForest(rf);
	        
	    final BasicTask basicTask = new BasicTask(marioAIOptions);
	        for (int i = 0; i < 10000; ++i)
	        {
	        	
	    	    basicTask.setOptionsAndReset(marioAIOptions);
	    	    if(i < 9000){
	    	    	marioAIOptions.setVisualization(false);
	    	    }
	    	    else{

	    	    	marioAIOptions.setVisualization(true);
	    	    	
	    	    }
	            int seed = 0;
	            do
	            {
	            	if(recordingMode){
	            		marioAIOptions.setAgent(agent2);
	            		
	            	}
	            	else{
	            		marioAIOptions.setAgent(agent);
	            	}
	            	
	                marioAIOptions.setLevelDifficulty(i);
	                marioAIOptions.setLevelRandSeed(seed++);
	                basicTask.runSingleEpisode(1);
	                basicTask.doEpisodes(1,true,1);
	                System.out.println(basicTask.getEnvironment().getEvaluationInfoAsString());
	            } while (basicTask.getEnvironment().getEvaluationInfo().marioStatus != Environment.MARIO_STATUS_WIN);
	        }
	//
	    System.exit(0);
	}

}