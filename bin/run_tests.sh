#!/bin/bash

# Initialize the `google` flag with default value
# if not included in script call, it will be blank
google=false
# Parse command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        -g|--enable-google)
            google=true
            ;;
        *)
            # Handle unknown arguments if needed
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done
# Check if the flag is set
if [ "$google" = true ]; then
    googlePythonStr="--enable-google"
else
    googlePythonStr=""
fi

timestr=$(date '+%Y-%m-%d_%H-%M-%S')
logdir=$SMARTHOME_ROOT/logs/run_tests/$timestr
mkdir -p $logdir

######### SAGE GPT 4 ##############
python $SMARTHOME_ROOT/sage/testing/test_runner.py --llm-type GPT --model-name gpt-4 --logpath journal_results/SAGE/GPT4/trial_1 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_gpt4_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --llm-type GPT --model-name gpt-4 --logpath journal_results/SAGE/GPT4/trial_2 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_gpt4_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --llm-type GPT --model-name gpt-4 --logpath journal_results/SAGE/GPT4/trial_3 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_gpt4_trial_3.txt

######### SAGE GPT 3.5 ##############
python $SMARTHOME_ROOT/sage/testing --llm-type GPT --model-name gpt-3.5-turbo --logpath journal_results/SAGE/GPT3.5/trial_1 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_gpt3.5_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --llm-type GPT --model-name gpt-3.5-turbo --logpath journal_results/SAGE/GPT3.5/trial_2 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_gpt3.5_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --llm-type GPT --model-name gpt-3.5-turbo --logpath journal_results/SAGE/GPT3.5/trial_3 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_gpt3.5_trial_3.txt

######### SAGE GPT 4 Turbo ##############
python $SMARTHOME_ROOT/sage/testing --llm-type GPT --model-name gpt-4-1106-preview --logpath journal_results/SAGE/GPT4-TURBO/trial_1 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_gpt4turbo_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --llm-type GPT --model-name gpt-4-1106-preview --logpath journal_results/SAGE/GPT4-TURBO/trial_2 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_gpt4turbo_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --llm-type GPT --model-name gpt-4-1106-preview --logpath journal_results/SAGE/GPT4-TURBO/trial_3 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_gpt4turbo_trial_3.txt

######### SAGE CLAUDE ##############
python $SMARTHOME_ROOT/sage/testing --llm-type CLAUDE --model-name claude-2.1 --logpath journal_results/SAGE/CLAUDE2.1/trial_1 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_claude2.1_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --llm-type CLAUDE --model-name claude-2.1 --logpath journal_results/SAGE/CLAUDE2.1/trial_2 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_claude2.1_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --llm-type CLAUDE --model-name claude-2.1 --logpath journal_results/SAGE/CLAUDE2.1/trial_3 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_claude2.1_trial_3.txt

######### SAGE LEMUR ##############
python $SMARTHOME_ROOT/sage/testing --llm-type LEMUR --model-name lemur --logpath journal_results/SAGE/LEMUR/trial_1 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_lemur_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --llm-type LEMUR --model-name lemur --logpath journal_results/SAGE/LEMUR/trial_2 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_lemur_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --llm-type LEMUR --model-name lemur --logpath journal_results/SAGE/LEMUR/trial_3 $googlePythonStr --test-scenario in-dist >> $logdir/log_sage_lemur_trial_3.txt



######### SASHA GPT 4 ##############
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type GPT --model-name gpt-4 --logpath journal_results/SASHA/GPT4/trial_1 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_gpt4_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type GPT --model-name gpt-4 --logpath journal_results/SASHA/GPT4/trial_2 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_gpt4_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type GPT --model-name gpt-4 --logpath journal_results/SASHA/GPT4/trial_3 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_gpt4_trial_3.txt

######### SASHA GPT 3.5 ##############
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type GPT --model-name gpt-3.5-turbo --logpath journal_results/SASHA/GPT3.5/trial_1 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_gpt3.5_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type GPT --model-name gpt-3.5-turbo --logpath journal_results/SASHA/GPT3.5/trial_2 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_gpt3.5_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type GPT --model-name gpt-3.5-turbo --logpath journal_results/SASHA/GPT3.5/trial_3 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_gpt3.5_trial_3.txt

######### SASHA GPT 4 Turbo ##############
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type GPT --model-name gpt-4-1106-preview --logpath journal_results/SASHA/GPT4-TURBO/trial_1 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_gpt4turbo_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type GPT --model-name gpt-4-1106-preview --logpath journal_results/SASHA/GPT4-TURBO/trial_2 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_gpt4turbo_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type GPT --model-name gpt-4-1106-preview --logpath journal_results/SASHA/GPT4-TURBO/trial_3 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_gpt4turbo_trial_3.txt

######### SASHA CLAUDE ##############
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type CLAUDE --model-name claude-2.1 --logpath journal_results/SASHA/CLAUDE2.1/trial_1 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_claude2.1_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type CLAUDE --model-name claude-2.1 --logpath journal_results/SASHA/CLAUDE2.1/trial_2 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_claude2.1_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type CLAUDE --model-name claude-2.1 --logpath journal_results/SASHA/CLAUDE2.1/trial_3 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_claude2.1_trial_3.txt

######### SASHA LEMUR ##############
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type LEMUR --model-name lemur --logpath journal_results/SASHA/LEMUR/trial_1 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_lemur_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type LEMUR --model-name lemur --logpath journal_results/SASHA/LEMUR/trial_2 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_lemur_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type SASHA --llm-type LEMUR --model-name lemur --logpath journal_results/SASHA/LEMUR/trial_3 $googlePythonStr --test-scenario in-dist >> $logdir/log_sasha_lemur_trial_3.txt




######### ZeroShot GPT 4 ##############
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type GPT --model-name gpt-4 --logpath journal_results/ZeroShot/GPT4/trial_1 --test-scenario in-dist >> $logdir/log_zeroshot_gpt4_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type GPT --model-name gpt-4 --logpath journal_results/ZeroShot/GPT4/trial_2 --test-scenario in-dist >> $logdir/log_zeroshot_gpt4_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type GPT --model-name gpt-4 --logpath journal_results/ZeroShot/GPT4/trial_3 --test-scenario in-dist >> $logdir/log_zeroshot_gpt4_trial_3.txt

######### ZeroShot GPT 3.5 ##############
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type GPT --model-name gpt-3.5-turbo --logpath journal_results/ZeroShot/GPT3.5/trial_1 --test-scenario in-dist >> $logdir/log_zeroshot_gpt3.5_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type GPT --model-name gpt-3.5-turbo --logpath journal_results/ZeroShot/GPT3.5/trial_2 --test-scenario in-dist >> $logdir/log_zeroshot_gpt3.5_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type GPT --model-name gpt-3.5-turbo --logpath journal_results/ZeroShot/GPT3.5/trial_3 --test-scenario in-dist >> $logdir/log_zeroshot_gpt3.5_trial_3.txt

######### ZeroShot GPT 4 Turbo ##############
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type GPT --model-name gpt-4-1106-preview --logpath journal_results/ZeroShot/GPT4-TURBO/trial_1 --test-scenario in-dist >> $logdir/log_zeroshot_gpt4turbo_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type GPT --model-name gpt-4-1106-preview --logpath journal_results/ZeroShot/GPT4-TURBO/trial_2 --test-scenario in-dist >> $logdir/log_zeroshot_gpt4turbo_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type GPT --model-name gpt-4-1106-preview --logpath journal_results/ZeroShot/GPT4-TURBO/trial_3 --test-scenario in-dist >> $logdir/log_zeroshot_gpt4turbo_trial_3.txt

######### ZeroShot CLAUDE ##############
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type CLAUDE --model-name claude-2.1 --logpath journal_results/ZeroShot/CLAUDE2.1/trial_1 --test-scenario in-dist >> $logdir/log_zeroshot_claude2.1_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type CLAUDE --model-name claude-2.1 --logpath journal_results/ZeroShot/CLAUDE2.1/trial_2 --test-scenario in-dist >> $logdir/log_zeroshot_claude2.1_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type CLAUDE --model-name claude-2.1 --logpath journal_results/ZeroShot/CLAUDE2.1/trial_3 --test-scenario in-dist >> $logdir/log_zeroshot_claude2.1_trial_3.txt

######### ZeroShot LEMUR ##############
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type LEMUR --model-name lemur --logpath journal_results/ZeroShot/LEMUR/trial_1 --test-scenario in-dist >> $logdir/log_zeroshot_lemur_trial_1.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type LEMUR --model-name lemur --logpath journal_results/ZeroShot/LEMUR/trial_2 --test-scenario in-dist >> $logdir/log_zeroshot_lemur_trial_2.txt
python $SMARTHOME_ROOT/sage/testing --coordinator-type ZEROSHOT --llm-type LEMUR --model-name lemur --logpath journal_results/ZeroShot/LEMUR/trial_3 --test-scenario in-dist >> $logdir/log_zeroshot_lemur_trial_3.txt
