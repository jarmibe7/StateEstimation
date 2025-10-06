# State Estimation Filters
### ME 469: HW0
### Author: Jared Berry
### Due: 10/05/2025

Hello!

To run for submission B, run the following command from directory HW0/:

```

python run.py

```

Plots for questions 2, 3, 6, 7, and 8 will populate in plots/, and data (ds0) is located in data/.

Additionally, measurements for question 6 will be printed to the terminal.

JSON files with metrics will also populate in metrics/

I've commented the commands for the extra plots I used in the report. This includes my plots for the
augmented UKF comparison, EKF, and particle filter. These will increase run time because I didn't have time
to optimize this code, but if you want to run them you can go to main() and uncomment the following lines:

```

694    # aug()

695    # study_comp()

696    # all_filters()

```