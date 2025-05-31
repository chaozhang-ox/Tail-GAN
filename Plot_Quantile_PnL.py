"""
Plot the quantiles of PnLs of benchmark strategies
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use("ggplot")

import numpy as np
import pandas as pd
import seaborn as sns
#sns.set()
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter

from TailGAN import *
from Dataset import Dataset_IS

stock_names =  ['Gaussian', r'AR(1) with $\phi_1=0.5$', r'AR(1) with $\phi_2=-0.12$', r'GARCH(1,1) with $t(5)$', r'GARCH(1,1) with $t(10)$']

your_path = 'your_path'  # Replace with your actual path
plot_path = join(your_path, 'Plots')
gen_data_path = join(your_path, 'Gens/')
result_data_path = join(your_path, 'Results/')


def Compute_PNL_Spc(R, opt):
    R_Tensor = Tensor(R)
    # Price
    prices_l = Inc2Price(R_Tensor)

    PNL_BH = BuyHold(prices_l, opt.Cap)
    PNL_l = [PNL_BH]

    thresholds_pct = [[31, 69]]
    for strategy in ['MR', 'TF']:
        if strategy == 'MR':
            for percentile_l in thresholds_pct:
                thresholds_array = gen_thresholds(opt.data_name, opt.tickers, strategy, percentile_l, 1000, opt.WH)
                PNL_MR = MeanRev(prices_l, opt.Cap, opt.WH, LR=opt.ratios[0], SR=opt.ratios[1],
                                 ST=thresholds_array[:, -1], LT=thresholds_array[:, -2])
                PNL_l.append(PNL_MR)
        elif strategy == 'TF':
            for percentile_l in thresholds_pct:
                thresholds_array = gen_thresholds(opt.data_name, opt.tickers, strategy, percentile_l, 1000, opt.WH)
                PNL_TF = TrendFollow(prices_l, opt.Cap, opt.WH, LR=opt.ratios[0], SR=opt.ratios[1],
                                     ST=thresholds_array[:, 0], LT=thresholds_array[:, 1])
                PNL_l.append(PNL_TF)
        else:
            pass

    PNL = torch.cat(PNL_l, dim=1)
    return PNL.cpu().numpy()


def Load_Data(opt):
    # Realistic Data
    dataset = Dataset_IS(tickers=opt.tickers, data_path=join(your_path, "gan_data", opt.data_name), length=opt.len)
    real_r = np.array([d.detach().numpy() for d in dataset.samples])
    sample_idx = random.sample(range(real_r.shape[0]), 10000)
    sample_real = real_r[sample_idx, :, :]
    R_dic = {'Real': sample_real}

    # Fake Data
    exps_l = os.listdir(result_data_path)
    exps_l.sort()

    save_path_wgan_l = [exp for exp in exps_l if ('WGAN' in exp and 'LR1e-06-1e-06' in exp)]
    save_path_p0_l = [exp for exp in exps_l if ('P0' in exp and 'Q5_' in exp)]
    save_path_p50_l = [exp for exp in exps_l if ('P50' in exp and 'MR' not in exp and 'Q5_' in exp)]
    save_path_p50_mrtf_R1_l = [exp for exp in exps_l if ('P50' in exp and 'MR' in exp and 'TF' in exp and 'Q5_' in exp)]
    # save_path_add_p50_mrtf_R1_l = [exp for exp in os.listdir(result_data_path) if ('P50' in exp and 'MR' in exp and 'MultiAdd' in exp)]

    category_dic = {
                    'Tail-GAN-Raw': save_path_p0_l,
                    'Tail-GAN-Static': save_path_p50_l,
                    'Tail-GAN': save_path_p50_mrtf_R1_l,
                    'WGAN': save_path_wgan_l,
    }

    for k in category_dic:
        this_version = category_dic[k][0]
        gen_data_path_spc = join(gen_data_path, 'gen_data_'+this_version)
        fake_l = []

        for epoch in range(2500, 3000):
            if isfile(join(gen_data_path_spc, 'Fake_id0_E%d.npy' % epoch)):
                tmp_fake = np.load(join(gen_data_path_spc, 'Fake_id0_E%d.npy' % epoch))
                fake_l.append(tmp_fake)

        fake_r = np.concatenate(fake_l)
        sample_idx = random.sample(range(fake_r.shape[0]), 10000)
        sample_fake_r = fake_r[sample_idx, :, :]
        R_dic[k] = sample_fake_r

    return R_dic


def VaR(PNL_dic, alpha):
    num_stocks = len(opt.tickers)
    stk_var_dic = {}
    mr_var_dic = {}
    tf_var_dic = {}
    for i in range(num_stocks):
        stk_var_dic[opt.tickers[i]] = []
        mr_var_dic[opt.tickers[i]] = []
        tf_var_dic[opt.tickers[i]] = []

        for j, k in enumerate(PNL_dic.keys()):
            PNL = PNL_dic[k]
            size = PNL.shape[0]
            data1_stock = np.sort(PNL[:, i])
            data1_mr = np.sort(PNL[:, num_stocks + i])
            data1_tf = np.sort(PNL[:, 2 * num_stocks + i])
            stk_var_dic[opt.tickers[i]].append(np.round(data1_stock[int(alpha*size)], 3))
            mr_var_dic[opt.tickers[i]].append(np.round(data1_mr[int(alpha*size)], 3))
            tf_var_dic[opt.tickers[i]].append(np.round(data1_tf[int(alpha*size)], 3))

    stk_var_df = pd.DataFrame(stk_var_dic).T
    stk_var_df.columns = PNL_dic.keys()
    mr_var_df = pd.DataFrame(mr_var_dic).T
    mr_var_df.columns = PNL_dic.keys()
    tf_var_df = pd.DataFrame(tf_var_dic).T
    tf_var_df.columns = PNL_dic.keys()

    stk_var_df.to_csv(join(plot_path, 'Stock_VaR_Synthetic.csv'))
    mr_var_df.to_csv(join(plot_path, 'MR_VaR_Synthetic.csv'))
    tf_var_df.to_csv(join(plot_path, 'TF_VaR_Synthetic.csv'))



def Plot_Rank(PNL_dic, opt):
    pdf_name = join(plot_path, 'Tails_IS_Synthetic.pdf')

    with PdfPages(pdf_name) as pdf:
        num_stocks = len(opt.tickers)
        fig, axes = plt.subplots(num_stocks, 3, figsize=(20, 24), sharex=True)

        cols = ['Static buy-and-hold', 'Mean-reversion', 'Trend-following']
        for ax, col in zip(axes[0], cols):
            ax.set_title(col, fontsize=16)

        rows = stock_names
        for ax, row in zip(axes[:, 0], rows):
            ax.set_ylabel(row+ '\n' + r'$\alpha$-quantile (log scale)', rotation=90, fontsize=16)

        columns = [r'$\alpha$ (log scale)'] * 3
        for ax, row in zip(axes[-1, :], columns):
            ax.set_xlabel(row, rotation=0, fontsize=16)

        for i in range(num_stocks):
            for j, k in enumerate(PNL_dic.keys()):
                PNL = PNL_dic[k]
                size = PNL.shape[0]
                data1_stock = PNL[:, i]
                data1_mr = PNL[:, num_stocks+i]
                data1_tf = PNL[:, 2*num_stocks+i]

                x = np.cumsum(np.ones(size)) / size
                axes[i, 0].grid(True)
                axes[i, 0].plot(x, np.sort(data1_stock), linewidth=3, label=k)
                axes[i, 0].set_yscale('symlog')
                axes[i, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

                axes[i, 1].grid(True)
                axes[i, 1].plot(x, np.sort(data1_mr), linewidth=3, label=k)
                axes[i, 1].set_yscale('symlog')
                axes[i, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

                axes[i, 2].grid(True)
                axes[i, 2].plot(x, np.sort(data1_tf), linewidth=3, label=k)
                axes[i, 2].set_yscale('symlog')
                axes[i, 2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            plt.xscale('log')

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.51, -0.00), fontsize=16)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()


if __name__ == '__main__':
    R_dic = Load_Data(opt)
    
    PNL_dic = {}
    
    for k in R_dic:
        PNL = Compute_PNL_Spc(R_dic[k], opt)
        if k == 'Real':
            k_name = 'Market Data'
        else:
            k_name = k
        PNL_dic[k_name] = PNL

    VaR(PNL_dic, alpha=0.05)
    Plot_Rank(PNL_dic, opt)