import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib
params = {'axes.titlesize':'18',
          'xtick.labelsize':'16',
          'ytick.labelsize':'16',
          'font.size':'19',
          'legend.fontsize':'medium',
          'lines.linewidth':'2.5',
          'font.weight':'normal',
          'lines.markersize':'14',
          'text.latex.preamble': r'\usepackage{amsfonts}',
          'lines.markerfacecolor':'none'
          }
matplotlib.rcParams.update(params)
plt.rcParams["mathtext.fontset"] = "cm"
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
markers = ['s', 'd', 'X', 'o', 'v', '*', '^', '8', 'h', '+']


def plot_results_example_audit(df: pd.DataFrame, lst_protocol: list, epsilon: float, k: int):
   
    df_eps = df.loc[(df.epsilon == epsilon) & (df.k == k)]
    dic_eps = df_eps.groupby('protocol')['eps_emp'].mean().to_dict()
    dic_stds = df_eps.groupby('protocol')['eps_emp'].std().to_dict()

    # Get empirical eps
    values = [dic_eps[key] for key in lst_protocol] 
    stds = [dic_stds[key] for key in lst_protocol]

    # Plotting
    plt.figure(figsize=(6, 3))
    plt.grid(color='grey', linestyle='dashdot', linewidth=0.5, zorder=0)
    plt.hlines(epsilon, -0.5, 7.5, label='Theoretical $\epsilon$', color ='red', linestyle='dashed')
    plt.bar(range(len(lst_protocol)), values, zorder=10, width=0.65)
    plt.xticks(range(len(lst_protocol)), lst_protocol, rotation = 45)
    plt.errorbar(range(len(lst_protocol)), values, yerr=stds, ecolor='black', capsize=5, zorder=50, linestyle='None')

    plt.ylabel('Estimated $\epsilon_{emp}$')    
    plt.xlabel('LDP Frequency Estimation Protocols')
    plt.savefig('results/fig_results_summary_audit.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)

    return plt.show()

def plot_results_pure_ldp_protocols(df: pd.DataFrame, analysis: str, lst_protocol: list, lst_eps: list, lst_k: list):
    fig, ax = plt.subplots(4, 2, figsize=(12, 14), sharey=True)
    plt.subplots_adjust(wspace=0.25, hspace=0.5)

    c = 0 # column
    for row, protocol in enumerate(lst_protocol):
        
        r = row // 2
        if c>1:
            c=0
        ax[r, c].yaxis.set_tick_params(which='both', labelbottom=True)
        ax[r, c].grid(color='grey', linestyle='dashdot', linewidth=0.5)
        ax[r, c].plot(lst_eps, label='Theoretical $\epsilon$', color ='black', linestyle='dashed')
        mkr_idx = 0
        for k in lst_k:

            results_k = []
            variation_k = []
            for epsilon in lst_eps:
                df_eps = df.loc[(df.protocol == protocol) & (df.epsilon == epsilon) & (df.k == k)]['eps_emp'].clip(0)
                results_k.append(df_eps.mean())
                variation_k.append(df_eps.std())
            
            std_minus = np.array(results_k) - np.array(variation_k)
            std_plus = np.array(results_k) + np.array(variation_k)        
            ax[r, c].fill_between(range(len(lst_eps)), std_minus, std_plus, alpha=0.3)
            ax[r, c].plot(results_k, label = 'k={}'.format(k), marker = markers[mkr_idx])
            
            mkr_idx+=1
        
        ax[r, c].set_yscale('log')
        ax[r, c].set_xticks(range(len(lst_eps)))
        ax[r, c].set_xticklabels(lst_eps)
        ax[r, c].set_title(protocol, fontsize=20)
        ax[r, c].set_ylabel('Estimated $\epsilon_{emp}$')
        ax[r, c].set_xlabel('Theoretical $\epsilon$')
        c += 1
        
    ax[0, 0].legend(columnspacing=0.8, ncol=8, loc='upper center', bbox_to_anchor=(1.05, 1.5))
    plt.savefig('results/fig_results_'+analysis+'.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)

    return plt.show()


def plot_results_approx_ldp_protocols(df: pd.DataFrame, analysis: str, lst_protocol: list, lst_eps: list, lst_k: list):

    fig, ax = plt.subplots(3, 2, figsize=(12, 10.5), sharey=True)
    plt.subplots_adjust(wspace=0.25, hspace=0.5)


    c = 0 # column
    for row, protocol in enumerate(lst_protocol):
        
        r = row // 2
        if c>1:
            c=0
        ax[r, c].yaxis.set_tick_params(which='both', labelbottom=True)
        ax[r, c].grid(color='grey', linestyle='dashdot', linewidth=0.5)
        if protocol != "GM":
            ax[r, c].plot(lst_eps, label='Theoretical $\epsilon$', color ='black', linestyle='dashed')
        else:
            ax[r, c].plot([eps for eps in lst_eps if eps <= 1], label='Theoretical $\epsilon$', color ='black', linestyle='dashed')

        mkr_idx = 0
        for k in lst_k:

            results_k = []
            variation_k = []
            for epsilon in lst_eps:
                df_eps = df.loc[(df.protocol == protocol) & (df.epsilon == epsilon) & (df.k == k)]['eps_emp'].clip(0)
                results_k.append(df_eps.mean())
                variation_k.append(df_eps.std())
            
            std_minus = np.array(results_k) - np.array(variation_k)
            std_plus = np.array(results_k) + np.array(variation_k)        
            ax[r, c].fill_between(range(len(lst_eps)), std_minus, std_plus, alpha=0.3)
            ax[r, c].plot(results_k, label = 'k={}'.format(k), marker = markers[mkr_idx])
            
            mkr_idx+=1

        ax[r, c].set_yscale('log')
        ax[r, c].set_xticks(range(len(lst_eps)))
        ax[r, c].set_xticklabels(lst_eps)
        ax[r, c].set_title(protocol + ', $\delta=1e^{-5}$', fontsize=20)
        ax[r, c].set_ylabel('Estimated $\epsilon_{emp}$')
        ax[r, c].set_xlabel('Theoretical $\epsilon$')
        ax[r, c].set_yticks([1e-1, 1e0, 1e1])  

        c += 1

        ax[0, 0].legend(columnspacing=0.3, ncol=8, loc='upper center', bbox_to_anchor=(1.05, 1.5))
        plt.savefig('results/fig_results_'+analysis+'.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)
        plt.show()

    return plt.show()


def plot_results_approx_ldp_delta_impact(df_delta_imp: pd.DataFrame, analysis: str, lst_protocol: list, lst_eps: list, lst_k: list, lst_delta: list):
    
    df_pure = pd.read_csv('results/ldp_audit_results_main_pure_ldp_protocols.csv')    
    df_approx = pd.read_csv('results/ldp_audit_results_main_approx_ldp_protocols.csv')
    df = pd.concat([df_pure, df_approx, df_delta_imp], axis=0)

    bar_width = 0.17  # width of bars
    eps_positions = np.arange(len(lst_eps))  # position of groups
    n_delta = len(lst_delta) + 1  # Include df_pure in the delta count

    for k in lst_k:
        print('--------------------------------k={}--------------------------------'.format(k))

        fig, ax = plt.subplots(3, 2, figsize=(10, 9), sharey=True)
        plt.subplots_adjust(wspace=0.3, hspace=0.6)
        for i, protocol in enumerate(lst_protocol):
            r, c = divmod(i, 2)  
            
            ax[r, c].yaxis.set_tick_params(which='both', labelbottom=True)
            ax[r, c].grid(color='grey', linestyle='dashdot', linewidth=0.5, zorder=0)

            # Adjust delta_positions to accommodate all bars, including the one for df_pure
            delta_positions = [eps_positions + (i - n_delta / 2) * bar_width for i in range(n_delta)]
            
            test_delta = []
            for delta_idx, delta in enumerate([None] + lst_delta):  # Add None for df_pure
                
                results_k = []
                variation_k = []
                for epsilon in lst_eps:
                    if delta_idx == 0:  # Handle df_pure
                        df_eps = df_pure.loc[(df_pure['protocol'] == protocol[1:]) & (df_pure['epsilon'] == epsilon) & (df_pure['k'] == k)]['eps_emp'].clip(0)
                        
                    else:
                        df_eps = df.loc[(df['protocol'] == protocol) & (df['epsilon'] == epsilon) & (df['delta'] == delta) & (df['k'] == k)]['eps_emp'].clip(0)
    
                    results_k.append(df_eps.mean() if df_eps.mean() > 0 else 0)
                    variation_k.append(df_eps.std() if df_eps.std() > 0 else 0)
                
                test_delta.append(results_k)
                label = '$\delta=0$' if delta_idx == 0 else f'$\delta=${"{:.0e}".format(delta)}'
                ax[r, c].bar(delta_positions[delta_idx], results_k, width=bar_width, label=label, zorder=5)
                ax[r, c].errorbar(delta_positions[delta_idx], results_k, yerr=variation_k, fmt='none', capsize=4, color='black', zorder=5)

            ax[r, c].set_xticks(eps_positions)
            ax[r, c].set_xticklabels(lst_eps)
            ax[r, c].set_yticks(lst_eps)
            ax[r, c].set_ylim(0, 1.05)  
            ax[r, c].set_title(protocol, fontsize=20)
            ax[r, c].set_ylabel('Estimated $\epsilon_{emp}$')
            ax[r, c].set_xlabel('Theoretical $\epsilon$')

        # Adjust legend for the entire figure, considering the extra "Pure" category
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, columnspacing=0.8, handlelength=1.5, loc='upper center', bbox_to_anchor=(0.49, 1.0), ncol=n_delta)
        plt.savefig('results/fig_results_'+analysis+'_k_'+str(k)+'.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)
    
    return plt.show()


def plot_results_lho_protocol(df_lho, analysis, lst_k, lst_g):
    
    plt.figure(figsize=(6, 3))
    plt.grid(color='grey', linestyle='dashdot', linewidth=0.5)
    mkr_idx = 0
    for k in lst_k:
        results_k = []
        variation_k = []
        for g in lst_g:
            df_eps = df_lho.loc[(df_lho.g == g) & (df_lho.k == k)]['eps_emp'].clip(0)
            results_k.append(df_eps.mean())
            variation_k.append(df_eps.std())

        std_minus = np.array(results_k) - np.array(variation_k)
        std_plus = np.array(results_k) + np.array(variation_k)        
        plt.fill_between(range(len(lst_g)), std_minus, std_plus, alpha=0.3)
        plt.plot(results_k, label = 'k={}'.format(k), marker = markers[mkr_idx])

        mkr_idx+=1

    plt.xticks(range(len(lst_g)), lst_g)
    plt.ylabel('Estimated $\epsilon_{emp}$')    

    plt.xlabel('Hash domain $g$')
    plt.legend(columnspacing=0.8, ncol=3, loc='upper center', bbox_to_anchor=(0.49, 1.48))
    plt.savefig('results/fig_results_'+analysis+'.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)

    return plt.show()


def plot_results_longitudinal_pure_ldp_protocols(df_pure_long: pd.DataFrame, df_approx_long: pd.DataFrame, pure_ldp_protocols: list, approx_lst_protocol: list, 
                                                 analysis: str, lst_eps: list, lst_k: list, lst_tau: list, eps_ub: float):
    
    df_seq = pd.concat([df_pure_long, df_approx_long], axis=0)
    lst_protocol = pure_ldp_protocols + approx_lst_protocol

    for k in lst_k:
        print('--------------------------------k={}--------------------------------'.format(k))
        
        fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
        plt.subplots_adjust(wspace=0.2, hspace=0.45)

        c = 0 # column
        for row, epsilon in enumerate(lst_eps):
            
            r = row // 2
            if c>1:
                c=0
            
            ax[r, c].yaxis.set_tick_params(which='both', labelbottom=True)
            ax[r, c].grid(color='grey', linestyle='dashdot', linewidth=0.5)
            ax[r, c].plot(epsilon*lst_tau, label='Theoretical $\\tau\epsilon$ (Sequential Composition)', color ='black', linestyle='dashed')
            ax[r, c].hlines(y=eps_ub, xmin=0, xmax=len(lst_tau)-1, label='Optimal $\epsilon_{OPT}$ (Monte Carlo Upper Bound)', color='red', linestyle='dotted')

            mkr_idx = 0
            for protocol in lst_protocol:

                results_k = []
                variation_k = []
                for idx_tau, tau in enumerate(lst_tau):
                    df_eps = df_seq.loc[(df_seq.protocol == protocol) & (df_seq.tau == tau) & (df_seq.k == k) & (df_seq.epsilon == epsilon)]['eps_emp'].clip(0)
    
                    # # turnaround missing exp
                    # if df_eps.shape[0] == 0:
                    #     if protocol in dic_missing.keys():
                    #         if protocol in lst_protocol_pure:
                    #             protocol_tst = 'THE'
                    #             df_eps = df_seq.loc[(df_seq.protocol == protocol_tst) & (df_seq.tau == tau) & (df_seq.k == k) & (df_seq.epsilon == epsilon)]['eps_emp'].clip(0)
                    #             results_k.append(df_eps.mean() * 0.95)
                    #             variation_k.append(df_eps.std())
                                                            
                    #         else:
                    #             protocol_tst = 'GM'
                    #             df_eps = df_seq.loc[(df_seq.protocol == protocol_tst) & (df_seq.tau == tau) & (df_seq.k == k) & (df_seq.epsilon == epsilon)]['eps_emp'].clip(0)
                    #             results_k.append(df_eps.mean() * np.array(dic_diff['AGM'][epsilon][idx_tau]) / np.array(dic_diff['GM'][epsilon][idx_tau]))
                    #             variation_k.append(df_eps.std())
                    # else:
                    results_k.append(df_eps.mean())
                    variation_k.append(df_eps.std())
                    
                std_minus = np.array(results_k) - np.array(variation_k)
                std_plus = np.array(results_k) + np.array(variation_k)        
                ax[r, c].fill_between(range(len(lst_tau)), std_minus, std_plus, alpha=0.3)
                ax[r, c].plot(results_k, label = protocol, marker = markers[mkr_idx])

                mkr_idx+=1

            ax[r, c].set_yscale('log')
            ax[r, c].set_xticks(range(len(lst_tau)))
            ax[r, c].set_xticklabels(lst_tau)
            ax[r, c].set_title('Per report $\epsilon={}$'.format(epsilon), fontsize=20)
            ax[r, c].set_ylabel('Estimated $\epsilon_{emp}$')
            ax[r, c].set_xlabel('Number of Data Collections $\\tau$')
            c += 1

        ax[0, 0].legend(columnspacing=0.3, handlelength=1.5, ncol=6, loc='upper center', bbox_to_anchor=(1.03, 1.55))
        plt.savefig('results/fig_results_'+analysis+'_k_'+str(k)+'.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)
        plt.show()

    return plt.show()

def plot_results_multidimensional(df: pd.DataFrame, lst_protocol: list, analysis: str, lst_eps: list, lst_k: list, lst_d: list):

    for k in lst_k:
        print('--------------------------------k={}--------------------------------'.format(k))

        fig, ax = plt.subplots(1, 2, figsize=(11.4, 2.8), sharey=True)
        plt.subplots_adjust(wspace=0.25, hspace=0.2)
        
        for r, d in enumerate(lst_d):
            
            ax[r].yaxis.set_tick_params(which='both', labelbottom=True)
            ax[r].grid(color='grey', linestyle='dashdot', linewidth=0.5)
            ax[r].plot(lst_eps, label='Theoretical $\epsilon$', color ='black', linestyle='dashed')

            mkr_idx=0
            for protocol in lst_protocol:
                results_k = []
                variation_k = []
                for epsilon in lst_eps:
                    df_eps = df.loc[(df.protocol == protocol) & (df.k == k) & (df.epsilon == epsilon) & (df.d == d)]['eps_emp'].clip(0)
                    results_k.append(df_eps.mean())
                    variation_k.append(df_eps.std())

                std_minus = np.array(results_k) - np.array(variation_k)
                std_plus = np.array(results_k) + np.array(variation_k)        
                ax[r].fill_between(range(len(lst_eps)), std_minus, std_plus, alpha=0.3)
                ax[r].plot(results_k, label = lst_protocol[mkr_idx], marker = markers[mkr_idx])
                mkr_idx+=1 

            ax[r].set_yscale('log')
            ax[r].set_xticks(range(len(lst_eps)))
            ax[r].set_xticklabels(lst_eps)

            ax[r].set_title('$d={}$'.format(d))
            ax[r].set_ylabel('Estimated $\epsilon_{emp}$')
            ax[r].set_xlabel('Theoretical $\epsilon$')

        ax[0].legend(columnspacing=0.3, ncol=3, loc='upper center', bbox_to_anchor=(1.1, 1.65))
        plt.savefig('results/fig_results_'+analysis+'_k_'+str(k)+'.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)
        plt.show()

    return plt.show()


def plot_result_pure_ldp_package(df: pd.DataFrame, analysis: str, lst_protocol: list, lst_eps: list, lst_k: list):

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 2.8))
    plt.subplots_adjust(wspace=0.25, hspace=0.2)

    # Define the coordinates for the box
    box_x = -0.22  # x-coordinate of the box
    box_y = .23  # y-coordinate of the box
    box_width = 1.6  # width of the box
    box_height = 0.9  # height of the box

    for r, protocol in enumerate(lst_protocol):
        
        ax[r].grid(color='grey', linestyle='dashdot', linewidth=0.5)
        ax[r].plot(lst_eps, label='Theoretical $\epsilon$', color ='black', linestyle='dashed')
        mkr_idx = 0
        for k in lst_k:
            results_k = []
            variation_k = []
            for epsilon in lst_eps:
                df_eps = df.loc[(df.protocol == protocol) & (df.epsilon == epsilon) & (df.k == k)]['eps_emp'].clip(0)
                results_k.append(df_eps.mean())
                variation_k.append(df_eps.std())
            
            std_minus = np.array(results_k) - np.array(variation_k)
            std_plus = np.array(results_k) + np.array(variation_k)        
            ax[r].fill_between(range(len(lst_eps)), std_minus, std_plus, alpha=0.3)
            ax[r].plot(results_k, label = 'k={}'.format(k), marker = markers[mkr_idx])
            mkr_idx+=1
        
        ax[r].set_yscale('log')
        ax[r].set_xticks(range(len(lst_eps)))
        ax[r].set_xticklabels(lst_eps)
        
        ax[r].set_title('SUE' if 'SUE' in protocol else 'OUE')
        ax[r].set_ylabel('Estimated $\epsilon_{emp}$')
        
        # Add the box to the plot
        box = Rectangle((box_x, box_y), box_width, box_height, fill=False, edgecolor='chocolate', linewidth=2)
        ax[r].add_patch(box)
        ax[r].set_xticklabels(lst_eps)
        ax[r].set_xlabel('Theoretical $\epsilon$')

    ax[0].legend(columnspacing=0.3, ncol=53, loc='upper center', bbox_to_anchor=(1.05, 1.45))
    plt.savefig('results/fig_results_'+analysis+'.pdf', dpi=500, bbox_inches = 'tight',pad_inches = 0.1)
    
    return plt.show()