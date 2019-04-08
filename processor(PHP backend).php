<?php

/**
The following methods when used with predictor methods will
support Regression in the php machine learning backend layer **/ 

namespace mlbackend_php;

defined('MOODLE_INTERNAL') || die();

use Phpml\Preprocessing\Normalizer;
use Phpml\CrossValidation\RandomSplit;
use Phpml\Dataset\ArrayDataset;
use Phpml\ModelManager;
use Phpml\Exception\InvalidArgumentException;


    /**
     * Train this processor regression model using the provided supervised learning dataset.
     *
     * @param string $uniqueid
     * @param \stored_file $dataset
     * @param string $outputdir
     * @return \stdClass
     */
    public function train_regression($uniqueid, \stored_file $dataset, $outputdir) {
        $modelfilepath = $this->get_model_filepath($outputdir);

        $modelmanager = new ModelManager();
        
      
        if (file_exists($modelfilepath)) {
            $regressor = $modelmanager->restoreFromFile($modelfilepath);//Load an existing model or
        } else {							//create a new model
            $regressor = new \Phpml\Regression\LeastSquares(); 			
        }

        $fh = $dataset->get_content_file_handle();

        // The first lines are var names and the second one values.
        $metadata = $this->extract_metadata($fh);

        // Skip headers.
        fgets($fh);

        $samples = array();
        $targets = array();
        while (($data = fgetcsv($fh)) !== false) {
            $sampledata = array_map('floatval', $data);
            $samples[] = array_slice($sampledata, 0, $metadata['nfeatures']);
            $targets[] = intval($data[$metadata['nfeatures']]);
                
            if (empty($morethan1sample) && $nsamples > 1) {
                $morethan1sample = true;
            }
        }
        fclose($fh);

        if (empty($morethan1sample)) {
            $resultobj = new \stdClass();
            $resultobj->status = \core_analytics\model::NO_DATASET;
            $resultobj->info = array();
            return $resultobj;
        }

       	//use train function to train the model. No option of partial train available for regression algorithms as yet.	
        $regressor->train($samples, $targets);

        $resultobj = new \stdClass();
        $resultobj->status = \core_analytics\model::OK;
        $resultobj->info = array();

        // Store the trained model.
        $modelmanager->saveToFile($regressor, $modelfilepath);

        return $resultobj;
    }

    /**
     * Estimates linear values for the provided dataset samples.
     *
     * @param string $uniqueid
     * @param \stored_file $dataset
     * @param mixed $outputdir
     * @return void
     */
    public function estimate($uniqueid, \stored_file $dataset, $outputdir) {
        
        $modelfilepath = $this->get_model_filepath($outputdir);

	//if model does not exist in the given path, throw an exception
        if (!file_exists($modelfilepath)) {
            throw new \moodle_exception('errorcantloadmodel', 'mlbackend_php', '', $modelfilepath);
        }

        $modelmanager = new ModelManager();
        $regressor = $modelmanager->restoreFromFile($modelfilepath);

        $fh = $dataset->get_content_file_handle();

        // The first lines are var names and the second one values.
        $metadata = $this->extract_metadata($fh);

        // Skip headers.
        fgets($fh);

        $sampleids = array();
        $samples = array();
        $predictions = array();
        while (($data = fgetcsv($fh)) !== false) {
            $sampledata = array_map('floatval', $data);
            $sampleids[] = $data[0];
            $samples[] = array_slice($sampledata, 1, $metadata['nfeatures']);
        }
        fclose($fh);

        $resultobj = new \stdClass();
        $resultobj->status = \core_analytics\model::OK;
        $resultobj->info = array();
	
	
        // Append predictions incrementally, we want $sampleids keys in sync with $predictions keys.
        $newpredictions = $regressor->predict($samples);
        foreach ($newpredictions as $prediction) {
            array_push($predictions, $prediction);
        }
       
	foreach ($predictions as $index => $prediction) {
            $resultobj->predictions[$index] = array($sampleids[$index], $prediction);
        }

        return $resultobj;
    }

    }

    /**
     * Evaluates this processor regression model using the provided supervised learning dataset.
     *
     * @throws new \coding_exception
     * @param string $uniqueid
     * @param float $maxdeviation
     * @param int $niterations
     * @param \stored_file $dataset
     * @param string $outputdir
     * 
     */
    public function evaluate_regression($uniqueid, $maxdeviation, $niterations, \stored_file $dataset, $outputdir) {
        $fh = $dataset->get_content_file_handle();

        // The first lines are var names and the second one values.
        $metadata = $this->extract_metadata($fh);

        // Skip headers.
        fgets($fh);

        if (empty($CFG->mlbackend_php_no_evaluation_limits)) {
            $samplessize = 0;
            $limit = get_real_size('500MB');

            // Just an approximation, will depend on PHP version, compile options...
            // Double size + zval struct (6 bytes + 8 bytes + 16 bytes) + array bucket (96 bytes)
            // https://nikic.github.io/2011/12/12/How-big-are-PHP-arrays-really-Hint-BIG.html.
            $floatsize = (PHP_INT_SIZE * 2) + 6 + 8 + 16 + 96;
        }

        $samples = array();
        $targets = array();
        while (($data = fgetcsv($fh)) !== false) {
            $sampledata = array_map('floatval', $data);

            $samples[] = array_slice($sampledata, 0, $metadata['nfeatures']);
            $targets[] = intval($data[$metadata['nfeatures']]);

            if (empty($CFG->mlbackend_php_no_evaluation_limits)) {
                // We allow admins to disable evaluation memory usage limits by modifying config.php.

                // We will have plenty of missing values in the dataset so it should be a conservative approximation.
                $samplessize = $samplessize + (count($sampledata) * $floatsize);

                // Stop fetching more samples.
                if ($samplessize >= $limit) {
                    $this->limitedsize = true;
                    break;
                }
            }
        }
        fclose($fh);
	
	$mse=0;
	
	 // Evaluate the model multiple times to confirm the results are not significantly random due to a short amount of data.
        for ($i = 0; $i < $niterations; $i++) {

            $regressor = new \Phpml\Regression\LeastSquares();

            // Split up the dataset in training and testing.
            $data = new RandomSplit(new ArrayDataset($samples, $targets), 0.2);

            $regressor->train($data->getTrainSamples(), $data->getTrainLabels());

            $predictedlabels = $regressor->predict($data->getTestSamples()); 
	    $predicttrain=$regressor->($data->getTrainSamples()); 
	    // mean square error of each iteration when comparing test data (stores test loss)
            $msetest = $msetest + $this->get_mse($data->getTestLabels(), $predictedlabels);  
	    // mean square error of each iteration when comparing training data (stores train loss)
	    $msetrain = $msetrain + $this->get_mse($data->getTrainLabels(), $predicttrain);  //stores train loss
	}
	
	$testloss=$msetest/$niterations;
	$trainloss=$msetrain/$niterations;

        return $this->get_evaluation_result_object_regressor($dataset, $testloss,$trainloss, $maxdeviation, $niterations);
	

    }

     /**
     * Returns the results objects from all evaluations.
     * 
     * @param \stored_file $dataset
     * @param float $testloss
     * @param float $trainloss
     * @param float $maxdeviation
     * @return \stdClass
     */

     protected function get_evaluation_result_object_regressor(\stored_file $dataset, $testloss,$trainloss, $maxdeviation,) {

        
        $modeldev = $testloss;

        // Let's fill the results object.
        $resultobj = new \stdClass();

        // Zero is ok, now we add other bits if something is not right.
        $resultobj->status = \core_analytics\model::OK;
        $resultobj->info = array();

        
        $resultobj->testloss = $testloss;
	$resultobj->trainloss = $trainloss;

        // If each iteration results varied too much we need more data to confirm that this is a valid model.
        if ($modeldev > $maxdeviation) {
            $resultobj->status = $resultobj->status + \core_analytics\model::NOT_ENOUGH_DATA;
            $a = new \stdClass();
            $a->deviation = $modeldev;
            $a->accepteddeviation = $maxdeviation;
            $resultobj->info[] = get_string('errornotenoughdatadev', 'mlbackend_php', $a);
        }


        return $resultobj;
    }


   
    
   
   

    /* Returns the mean square error in the prediction
     * @param array $testlabels
     * @param array $predictedlabels
     *
     * @return float
     *
     * @throws InvalidArgumentException
     */

    protected function get_mse($testlabels, $predictedlabels) {

	if (count($testlabels) !== count($predictedlabels)) {
           throw InvalidArgumentException::arraySizeNotMatch();
           }

	$mse = 0;
 	$count = count($testlabels);

	for ($i = 0; $i < $count; ++$i) {
            $mse += abs($testlabels[$i] - $predictedlabels[$i]) ** 2;
            }
	$rmse=$rmse/$count;
	
     

	return $rmse;
    }

}
