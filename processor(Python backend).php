<?php
/** The following functions whenimplemented with the predictor class
will add regression cpabilities to the python machine learning backend. **/

namespace mlbackend_python;

defined('MOODLE_INTERNAL') || die();


    /**
     * Exports the machine learning model.
     *
     * @throws \moodle_exception
     * @param  string $uniqueid  The model unique id
     * @param  string $modeldir  The directory that contains the trained model.
     * @param  string $type  Classifier or Regressor.
     * @return string            The path to the directory that contains the exported model.
     */
    public function export(string $uniqueid, string $modeldir, string $type) : string {

        // We include an exporttmpdir as we want to be sure that the file is not deleted after the
        // python process finishes.
        $exporttmpdir = make_request_directory('mlbackend_python_export');

        $cmd = "{$this->pathtopython} -m moodlemlbackend.export " .
            escapeshellarg($uniqueid) . ' ' .
            escapeshellarg($modeldir) . ' ' .
            escapeshellarg($exporttmpdir) . ' ' .
	    escapeshellarg($type);

        if (!PHPUNIT_TEST && CLI_SCRIPT) {
            debugging($cmd, DEBUG_DEVELOPER);
        }

        $output = null;
        $exitcode = null;
        $exportdir = exec($cmd, $output, $exitcode);

        if ($exitcode != 0) {
            throw new \moodle_exception('errorexportmodelresult', 'analytics');
        }

        if (!$exportdir) {
            throw new \moodle_exception('errorexportmodelresult', 'analytics');
        }

        return $exportdir;
    }

    /**
     * Imports the provided machine learning model.
     *
     * @param  string $uniqueid The model unique id
     * @param  string $modeldir  The directory that will contain the trained model.
     * @param  string $importdir The directory that contains the files to import.
     * @param  string $type  Classifier or Regressor.
     * @return bool Success
     */
    public function import(string $uniqueid, string $modeldir, string $importdir, string $type) : bool {

        $cmd = "{$this->pathtopython} -m moodlemlbackend.import " .
            escapeshellarg($uniqueid) . ' ' .
            escapeshellarg($modeldir) . ' ' .
            escapeshellarg($importdir) . ' ' .
	    escapeshellarg($type);

        if (!PHPUNIT_TEST && CLI_SCRIPT) {
            debugging($cmd, DEBUG_DEVELOPER);
        }

        $output = null;
        $exitcode = null;
        $success = exec($cmd, $output, $exitcode);

        if ($exitcode != 0) {
            throw new \moodle_exception('errorimportmodelresult', 'analytics');
        }

        if (!$success) {
            throw new \moodle_exception('errorimportmodelresult', 'analytics');
        }

        return $success;
    }

    /**
     * Train this processor regression model using the provided supervised learning dataset.
     *
     * @throws new \coding_exception
     * @param string $uniqueid
     * @param \stored_file $dataset
     * @param string $outputdir
     * @return \stdClass
     */
    public function train_regression($uniqueid, \stored_file $dataset, $outputdir) {
        // Obtain the physical route to the file.
        $datasetpath = $this->get_file_path($dataset);

        $cmd = "{$this->pathtopython} -m moodlemlbackend.regressortraining " .
            escapeshellarg($uniqueid) . ' ' .
            escapeshellarg($outputdir) . ' ' .
            escapeshellarg($datasetpath);

        if (!PHPUNIT_TEST && CLI_SCRIPT) {
            debugging($cmd, DEBUG_DEVELOPER);
        }

        $output = null;
        $exitcode = null;
        $result = exec($cmd, $output, $exitcode);

        if (!$result) {
            throw new \moodle_exception('errornopredictresults', 'analytics');
        }

        if (!$resultobj = json_decode($result)) {
            throw new \moodle_exception('errorpredictwrongformat', 'analytics', '', json_last_error_msg());
        }

        if ($exitcode != 0) {
            if (!empty($resultobj->errors)) {
                $errors = $resultobj->errors;
                if (is_array($errors)) {
                    $errors = implode(', ', $errors);
                }
            } else if (!empty($resultobj->info)) {
                // Show info if no errors are returned.
                $errors = $resultobj->info;
                if (is_array($errors)) {
                    $errors = implode(', ', $errors);
                }
            }
            $resultobj->info = array(get_string('errorpredictionsprocessor', 'analytics', $errors));
        }

        return $resultobj;;
    }

    /**
     * Estimates linear values for the provided dataset samples.
     *
     * @throws new \coding_exception
     * @param string $uniqueid
     * @param \stored_file $dataset
     * @param mixed $outputdir
     * @return void
     */
    public function estimate($uniqueid, \stored_file $dataset, $outputdir) {
         // Obtain the physical route to the file.
        $datasetpath = $this->get_file_path($dataset);

        $cmd = "{$this->pathtopython} -m moodlemlbackend.regressorprediction " .
            escapeshellarg($uniqueid) . ' ' .
            escapeshellarg($outputdir) . ' ' .
            escapeshellarg($datasetpath);

        if (!PHPUNIT_TEST && CLI_SCRIPT) {
            debugging($cmd, DEBUG_DEVELOPER);
        }

        $output = null;
        $exitcode = null;
        $result = exec($cmd, $output, $exitcode);

        if (!$result) {
            throw new \moodle_exception('errornopredictresults', 'analytics');
        }

        if (!$resultobj = json_decode($result)) {
            throw new \moodle_exception('errorpredictwrongformat', 'analytics', '', json_last_error_msg());
        }

        if ($exitcode != 0) {
            if (!empty($resultobj->errors)) {
                $errors = $resultobj->errors;
                if (is_array($errors)) {
                    $errors = implode(', ', $errors);
                }
            } else if (!empty($resultobj->info)) {
                // Show info if no errors are returned.
                $errors = $resultobj->info;
                if (is_array($errors)) {
                    $errors = implode(', ', $errors);
                }
            }
            $resultobj->info = array(get_string('errorpredictionsprocessor', 'analytics', $errors));
        }

        return $resultobj;
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
     * @return \stdClass
     */
    public function evaluate_regression($uniqueid, $maxdeviation, $niterations, \stored_file $dataset, $outputdir) {
       // Obtain the physical route to the file.
        $datasetpath = $this->get_file_path($dataset);

        $cmd = "{$this->pathtopython} -m moodlemlbackend.regressorevaluation " .
            escapeshellarg($uniqueid) . ' ' .
            escapeshellarg($outputdir) . ' ' .
            escapeshellarg($datasetpath) . ' ' .
            escapeshellarg($maxdeviation) . ' ' .
            escapeshellarg($niterations);

        if (!PHPUNIT_TEST && CLI_SCRIPT) {
            debugging($cmd, DEBUG_DEVELOPER);
        }

        $output = null;
        $exitcode = null;
        $result = exec($cmd, $output, $exitcode);

        if (!$result) {
            throw new \moodle_exception('errornopredictresults', 'analytics');
        }

        if (!$resultobj = json_decode($result)) {
            throw new \moodle_exception('errorpredictwrongformat', 'analytics', '', json_last_error_msg());
        }

        return $resultobj;
    }


