# Tail-GAN
code scripts for Tail-GAN: Learning to Simulate Tail Risk Scenarios

1. TailGAN.py is the main file to train the model.
2. Tranform.py includes the functions to convert the true/generated returns to PnLs of strategies that users are interested.
3. gen_threshols.py is used to compute the thresholds to trigger the trading signals for mean-reversion or trend-following strategies.
4. util.py is the Neural Sorting module.
