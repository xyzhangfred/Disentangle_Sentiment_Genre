import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import numpy as np
import os, json

def plot_results_DEBert( train_losses, val_losses, transfer_results,save_dir,figname, finetune_results = None, model_type = 'TripletBert',eval_ratios = [0.05,0.25,0.5,0.75,0.95], show = True):
    
    val_ratio_num = len(val_losses['0'])
    ratio_ids = list(range(val_ratio_num))
    
    fig = plt.figure(constrained_layout=True, figsize=(25,20))
    gs = fig.add_gridspec(6, len(ratio_ids))

    steps = [int(k) for k in train_losses.keys()]
    steps.sort()

    ax1 = fig.add_subplot(gs[0,:])

    ax1.set_title('Training Loss')
    if model_type == 'TripletBert':
        ax1.plot(steps, [train_losses[str(s)][0]  for s in steps],'r-',label = 'total_loss')
        ax1.plot(steps, [train_losses[str(s)][1]  for s in steps],'b-', label = 'loss_sent')
        ax1.plot(steps, [train_losses[str(s)][2]  for s in steps],'r--', label = 'loss_other')
    elif model_type == 'AdvBert':
        ax1.plot(steps, [train_losses[str(s)][1]  for s in steps],'b-', label = 'loss_sent')
        ax1.plot(steps, [train_losses[str(s)][0]  for s in steps],'r-',label = 'loss_other')
        ax1.plot(steps, [train_losses[str(s)][2]  for s in steps],'b--', label = 'dis_loss_sent')
        ax1.plot(steps, [train_losses[str(s)][3]  for s in steps],'r--', label = 'dis_loss_other')

    ax1.legend()

    steps = [int(k) for k in val_losses.keys()]
    steps.sort()

    for i, rat in enumerate(ratio_ids):
        if i == 0:
            ax_0 = fig.add_subplot(gs[1,i])
            ax = ax_0
        else:
            ax = fig.add_subplot(gs[1,i], sharey = ax_0)            
        ax.set_title('val_loss on ratio_id = '+str(eval_ratios[rat]))

        if model_type == 'TripletBert':

            ax.plot(steps, [val_losses[str(s)][i][0]  for s in steps],'r-', label = 'total_loss')
            ax.plot(steps, [val_losses[str(s)][i][1]  for s in steps],'g-', label = 'loss_sent')
            ax.plot(steps, [val_losses[str(s)][i][2]  for s in steps],'b-', label = 'loss_other')
        elif model_type == 'AdvBert':
            ax.plot(steps, [val_losses[str(s)][i][0]  for s in steps],'b-', label = 'loss_sent')
            ax.plot(steps, [val_losses[str(s)][i][1]  for s in steps],'r-',label = 'loss_other')
            ax.plot(steps, [val_losses[str(s)][i][2]  for s in steps],'b--', label = 'dis_loss_sent')
            ax.plot(steps, [val_losses[str(s)][i][3]  for s in steps],'r--', label = 'dis_loss_other')

        ax.legend()

    steps = [int(k) for k in transfer_results.keys()]
    steps.sort()

    ax2 = fig.add_subplot(gs[2,0:])
    ax2.set_title('||W_sent|| / ||W_other|| For Sentiment')
    ax2.axhline(y=1, ls = '--', c = 'k')
    ax2.plot(steps, [transfer_results[str(s)]['0']['combined'][1][0]  for s in steps],'b-', label = 'For Sentiment')
    ax2.legend()


    ax3 = fig.add_subplot(gs[3,0:])
    ax3.set_title('||W_sent|| / ||W_other|| For Genre')
    ax3.axhline(y=1, ls = '--', c = 'k')
    ax3.plot(steps, [transfer_results[str(s)]['0']['combined'][1][1]  for s in steps],'b-', label = 'For Genre')
    ax3.legend()


    for i, rat in enumerate(ratio_ids):
        if i == 0:
            ax_0 = fig.add_subplot(gs[4,i])
            ax = ax_0
        else:
            ax = fig.add_subplot(gs[4,i], sharey = ax_0)
        ax.set_title('Sentiment accuracy  on eval_ratio  = '+str(eval_ratios[rat]))
        ax.plot(steps, [transfer_results[str(s)][str(i)]['sent_enc'][0][0]  for s in steps],'r-', label = 'sent_enc')
        ax.plot(steps, [transfer_results[str(s)][str(i)]['other_enc'][0][0]  for s in steps],'g-', label = 'other_enc')
        ax.legend()

    for i, rat in enumerate(ratio_ids):
        if i == 0:
            ax_0 = fig.add_subplot(gs[5,i])
            ax = ax_0
        else:
            ax = fig.add_subplot(gs[5,i], sharey = ax_0)
        ax.set_title('Genre accuracy on eval_ratio  = '+str(eval_ratios[rat]))
        ax.plot(steps, [transfer_results[str(s)][str(i)]['sent_enc'][0][1]  for s in steps],'r-', label = 'sent_enc')
        ax.plot(steps, [transfer_results[str(s)][str(i)]['other_enc'][0][1] for s in steps],'g-', label = 'other_enc')
        ax.legend()

    # ax4.set_title('Sentiment/Genre accuracy on Source')
    # ax4.plot(steps, [transfer_accs[str(s)]['Sent_orig'][0]  for s in steps], 'r-' , label = 'Sent_Enc For Sentiment')
    # ax4.plot(steps, [transfer_accs[str(s)]['Sent_orig'][1]  for s in steps], 'g-' , label = 'Other_Enc For Sentiment')
    # # ax4.plot(steps, [transfer_accs[str(s)]['Genre'][2]  for s in steps] ,'b-' , label = 'Combined on Target')
    # ax4.plot(steps, [transfer_accs[str(s)]['Genre_orig'][0]  for s in steps], 'r--' , label = 'Sent_Enc For Genre')
    # ax4.plot(steps, [transfer_accs[str(s)]['Genre_orig'][1]  for s in steps], 'g--' , label = 'Other_Enc For Genre')
    # # ax4.plot(steps, [transfer_accs[str(s)]['Genre_orig'][2]  for s in steps], 'b--' , label = 'Combined on Source')
    # ax4.legend()

    # ax5.set_title('Weight ratio for Sentiment/Genre on Target')
    # ind = np.arange(len(steps))    # the x locations for the groups
    # width = 0.35
    # ax5.bar(ind - width/2, [weight_ratios[str(s)]['Sent']  for s in steps],  width, label='For Sentiment', tick_label = steps)
    # ax5.bar(ind + width/2, [weight_ratios[str(s)]['Genre']  for s in steps],  width, label='For Genre', tick_label = steps)
    # ax5.legend()


    # ax6.set_title('Weight ratio for Sentiment/Genre on Source')
    
    # ind = np.arange(len(steps))    # the x locations for the groups
    # width = 0.35
    # ax6.bar(ind - width/2, [weight_ratios[str(s)]['Sent_orig']  for s in steps],  width, label='For Sentiment', tick_label = steps)
    # ax6.bar(ind + width/2, [weight_ratios[str(s)]['Genre_orig']  for s in steps],  width, label='For Genre', tick_label = steps)
    # ax6.legend()
    
    plt.savefig(os.path.join(save_dir,figname))
    if show:
        plt.show()
    plt.close()

    if finetune_results is not None:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
        steps = list(finetune_results.keys())
        ax1.set_title('Sentiment accuracy  on validation sets with different ratios')
        ratios = [0.05, 0.25, 0.5, 0.75, 0.95]
        for i,r in enumerate(ratios):
            ax1.plot(steps, [finetune_results[str(s)][i][0]  for s in steps], label = 'r = '+str(r))
        ax1.legend()

        ax2.set_title('Genre accuracy  on validation sets with different ratios')
        ratios = [0.05, 0.25, 0.5, 0.75, 0.95]
        for i,r in enumerate(ratios):
            ax2.plot(steps, [finetune_results[str(s)][i][1]  for s in steps], label = 'r = '+str(r))
        ax2.legend()
        plt.savefig(os.path.join(save_dir,'finetune.png'))
        if show:
            plt.show()
        plt.close()



def plot_results_Bert( finetune_results,save_dir,  show = True, pred_both = True):
    steps = list(finetune_results.keys())
    if pred_both:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
        ax1.set_title('Sentiment accuracy  on validation sets with different ratios')
        ratios = [0.05, 0.25, 0.5, 0.75, 0.95]
        for i,r in enumerate(ratios):
            ax1.plot(steps, [finetune_results[str(s)][i][0]  for s in steps], label = 'r = '+str(r))
        ax1.legend()

        ax2.set_title('Genre accuracy  on validation sets with different ratios')
        ratios = [0.05, 0.25, 0.5, 0.75, 0.95]
        for i,r in enumerate(ratios):
            ax2.plot(steps, [finetune_results[str(s)][i][1]  for s in steps], label = 'r = '+str(r))
        ax2.legend()
    else:
        fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
        ax1.set_title('Sentiment accuracy  on validation sets with different ratios')
        ratios = [0.05, 0.25, 0.5, 0.75, 0.95]
        for i,r in enumerate(ratios):
            ax1.plot(steps, [finetune_results[str(s)][i]  for s in steps], label = 'r = '+str(r))
        ax1.legend()
    
    plt.savefig(os.path.join(save_dir,'finetune.png'))
    if show:
        plt.show()
    plt.close()

def compare_best_acc( res_dir = None ,save_dir = None, train_ratio = 0.05,eval_ratios = [0.05,0.25,0.5,0.75,0.95],num_train_datas = [50,100,200,400], strategies = ['only_diff', 'opposite', 'mixture','only_same'], show = True):
    best_sent_accs = {r:{n:{}  for n in num_train_datas} for r in eval_ratios}
    for n in num_train_datas:
        bert_results_sent= json.load(open( os.path.join(res_dir, 'finetune_result_train_ratio_{}_{}_num_labeled_{}'.format(train_ratio, 'pred_sent', n) +'/finetune_final_results.json') ))
        bert_results_sent = list(bert_results_sent.values())

        # bert_results_both = json.load(open( os.path.join(res_dir, 'finetune_result_train_ratio_{}_{}_num_labeled_{}'.format(train_ratio, 'pred_both', n) +'/finetune_final_results.json') ))
        # bert_results_both = list(bert_results_both.values())
        # bert_results_both = [[ll[0] for ll in l ] for l in bert_results_both ]
        for i,r in enumerate(eval_ratios):       
            bert_best_results_sent = max(b[i] for b in bert_results_sent)
            best_sent_accs[r][n]['bert_sent'] = bert_best_results_sent

            # bert_best_results_both = max(b[i] for b in bert_results_both)
            # best_sent_accs[r][n]['bert_both'] = bert_best_results_both
            for strategy in strategies:
                # for finetune_type in ['pred_sent', 'pred_both']:
                for finetune_type in ['pred_sent']:
                    outdir = os.path.join(res_dir,  'TripletBert_{}_{}_{}_num_labeled_{}'.format(train_ratio,strategy,finetune_type,n) )
                    results = json.load(open(outdir +'/transfer_results.json'))
                    results = [results[s] for s in results]
                    best_sent_accs[r][n][strategy +'_'+ finetune_type]= max( [ r[str(i)]['sent_enc'][0][0] for r in results])
    
    matplotlib.rcParams.update({'font.size': 35})

    for r in eval_ratios:
        figname = 'best_sent_acc_train_r_{}_eval_r_{}.png'.format(train_ratio, r)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
        ax.set_title('Train ratio = {}, eval ratio = {}'.format(train_ratio,r))
        ax.plot(num_train_datas, [best_sent_accs[r][n]['bert_sent']  for n in num_train_datas],'k:', label = 'bert_sent',linewidth=6.0)
        # ax.plot(num_train_datas, [best_sent_accs[r][n]['bert_both']  for n in num_train_datas],'b:', label = 'bert_both',linewidth=6.0)
       
        ax.set_xlabel('Num_training_data')
        ax.set_xticks(num_train_datas)
        ax.set_ylabel('Sentiment accuracy')
        colors = ['r','y','g', 'c']
        for i,strategy in enumerate(strategies):
            c = colors[i]
            ax.plot(num_train_datas, [best_sent_accs[r][n][strategy +'_'+ 'pred_sent']  for n in num_train_datas],c+'-', label = strategy +'_'+ 'pred_sent')
            ax.plot(num_train_datas, [best_sent_accs[r][n][strategy +'_'+ 'pred_both']  for n in num_train_datas],c+'-.', label = strategy +'_'+ 'pred_both')
        ax.legend()
        plt.savefig(os.path.join(save_dir,figname))


def compare_best_acc_across_reps( res_root_dir = None ,save_dir = None, train_ratios = [0.05,0.25,0.5],\
    eval_ratios = [0.05,0.25,0.5,0.75,0.95],num_train_datas = [25,50,100,200,400],rep_inds = [0,1], \
        strategies = ['only_diff', 'opposite', 'mixture','only_same'], show = True, multi_task = False):
    
    all_reps ={}
    for rep_i in rep_inds:
        best_sent_accs = {(t_r,e_r):{n:{}  for n in num_train_datas} for e_r in eval_ratios for t_r in train_ratios}
        for n in num_train_datas:
            for t_r in train_ratios:
                bert_results_sent= json.load(open( os.path.join(res_root_dir,'rep_'+str(rep_i), 'finetune_result_train_ratio_{}_{}_num_labeled_{}'.format(t_r, 'pred_sent', n) +'/finetune_final_results.json') ))
                bert_results_sent = list(bert_results_sent.values())

                bert_results_sent_twice= json.load(open( os.path.join(res_root_dir,'rep_'+str(rep_i), 'finetune_result_train_ratio_{}_{}_num_labeled_{}'.format(t_r, 'pred_sent', 2 * n) +'/finetune_final_results.json') ))
                bert_results_sent_twice = list(bert_results_sent_twice.values())
                if multi_task:
                    bert_results_both = json.load(open( os.path.join(res_root_dir,'rep_'+str(rep_i), 'finetune_result_train_ratio_{}_{}_num_labeled_{}'.format(t_r, 'pred_both', n) +'/finetune_final_results.json') ))
                    bert_results_both = list(bert_results_both.values())
                    bert_results_both = [[ll[0] for ll in l ] for l in bert_results_both ]
                for i,e_r in enumerate(eval_ratios):          
                    bert_best_results_sent = max(b[i] for b in bert_results_sent)
                    print ('bert_results_sent', bert_results_sent)
                    print ('bert_results_sent_twice', bert_results_sent_twice)
                    bert_best_results_sent_twice = max(b[i] for b in bert_results_sent_twice)

                    best_sent_accs[(t_r,e_r)][n]['bert_sent'] = bert_best_results_sent
                    best_sent_accs[(t_r,e_r)][n]['bert_sent_twice'] = bert_best_results_sent_twice
                    if multi_task:
                        bert_best_results_both = max([b[i] for b in bert_results_both])
                        best_sent_accs[(t_r,e_r)][n]['bert_both'] = bert_best_results_both

                    # bert_best_results_both = max(b[i] for b in bert_results_both)
                    # bert_best_results_both = [b[i] for b in bert_results_both][-1]
                    for strategy in strategies:
                        for finetune_type in ['pred_sent', 'pred_both']:
                            outdir = os.path.join(res_root_dir,'rep_'+str(rep_i),  'TripletBert_{}_{}_{}_num_labeled_{}'.format(t_r,strategy,finetune_type,n) )
                            results = json.load(open(outdir +'/transfer_results.json'))
                            results = [results[key] for key in results]
                            best_sent_accs[(t_r,e_r)][n][strategy +'_'+ finetune_type]= max([res[str(i)]['sent_enc'][0][0] for res in results])
        all_reps[rep_i] = best_sent_accs
    matplotlib.rcParams.update({'font.size': 25})
    final_res = {}
    err_bars = {}

    ##take average

    for t_r in train_ratios:
        for e_r in eval_ratios:
            final_res[(t_r,e_r)] = {}
            err_bars[(t_r,e_r)] = {}
            for n in num_train_datas:
                final_res[(t_r,e_r)][n] = {}
                err_bars[(t_r,e_r)][n] = {}
                keys = list(best_sent_accs[(t_r,e_r)][n].keys())
                for k in keys:
                    final_res[(t_r,e_r)][n][k] = 0
                    err_bars[(t_r,e_r)][n][k] = []
                    for rep_i in rep_inds:
                        final_res[(t_r,e_r)][n][k] += all_reps[rep_i][(t_r,e_r)][n][k]
                    
                    # max_val = max([ all_reps[rep_i][(t_r,e_r)][n][k] for rep_i in rep_inds ])
                    # min_val = min([ all_reps[rep_i][(t_r,e_r)][n][k] for rep_i in rep_inds ])
                    final_res[(t_r,e_r)][n][k] /= len(rep_inds)
                    # err_bars[(t_r,e_r)][n][k] = [final_res[(t_r,e_r)][n][k] - min_val, max_val - final_res[(t_r,e_r)][n][k]]
                    # err_bars[(t_r,e_r)][n][k] = 2.776 * ( np.std([ all_reps[rep_i][(t_r,e_r)][n][k] for rep_i in rep_inds ]) / np.sqrt(5))
                    # breakpoint()

    figname = 'best_sent_acc_all_rep_all_ratio.png'
    fig, axes = plt.subplots(nrows=len(train_ratios), ncols=len(eval_ratios), figsize=(35,35), sharey=True)
    for i,t_r in enumerate(train_ratios):
        for j,e_r in enumerate(eval_ratios):
            ax = axes[i,j]
            # print (np.asarray([err_bars[(t_r,e_r)][n]['bert_sent']  for n in num_train_datas]).T)
            ax.plot([2 * n for n in num_train_datas], [final_res[(t_r,e_r)][n]['bert_sent']  for n in num_train_datas],'k:',\
                label = 'bert_sent',linewidth=6.0)
            ax.plot([2 * n for n in num_train_datas], [final_res[(t_r,e_r)][n]['bert_sent_twice']  for n in num_train_datas],'g:',\
                label = 'bert_sent_twice',linewidth=6.0)
            ax.plot([2 * n for n in num_train_datas], [final_res[(t_r,e_r)][n]['bert_both']  for n in num_train_datas],'b:', label = 'bert_both',linewidth=6.0)
            # ax.set_xlabel('Num_training_data')
            ax.set_xticks(num_train_datas)
            # ax.set_ylabel('Sentiment accuracy')
            ax.set_title('Train {}, eval {}'.format(t_r,e_r))
            colors = ['r','y','g', 'c']
            handles, labels = ax.get_legend_handles_labels()
            for k,strategy in enumerate(strategies):
                c = colors[k]
                ax.plot([2 * n for n in num_train_datas], [final_res[(t_r,e_r)][n][strategy +'_'+ 'pred_sent']  for n in num_train_datas],c+'-', label = strategy +'_'+ 'pred_sent', \
                    )
                ax.plot([2 * n for n in num_train_datas], [final_res[(t_r,e_r)][n][strategy +'_'+ 'pred_both']  for n in num_train_datas],c+'-.', label = strategy +'_'+ 'pred_both', \
                    )
                handles, labels = ax.get_legend_handles_labels()
            # ax.legend()

    fontP = FontProperties()
    fontP.set_size('small')
    fig.legend(handles, labels, loc='lower right', prop = fontP)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir,figname))


def best_acc_all_reps( res_root_dir = None ,save_dir = None, train_ratios = [0.05,0.25,0.5],\
    eval_ratios = [0.05,0.25,0.5,0.75,0.95],num_train_datas = [25,50,100,200,400],rep_inds = [0,1], \
        strategies = ['only_diff', 'opposite', 'mixture','only_same'], show = True, multi_task = False):
    
    all_reps ={}
    for rep_i in rep_inds:
        best_sent_accs = {(t_r,e_r):{n:{}  for n in num_train_datas} for e_r in eval_ratios for t_r in train_ratios}
        for n in num_train_datas:
            for t_r in train_ratios:
                bert_results_sent= json.load(open( os.path.join(res_root_dir,'rep_'+str(rep_i), 'finetune_result_train_ratio_{}_{}_num_labeled_{}'.format(t_r, 'pred_sent', n) +'/finetune_final_results.json') ))
                bert_results_sent = list(bert_results_sent.values())

                bert_results_sent_twice= json.load(open( os.path.join(res_root_dir,'rep_'+str(rep_i), 'finetune_result_train_ratio_{}_{}_num_labeled_{}'.format(t_r, 'pred_sent', 2 * n) +'/finetune_final_results.json') ))
                bert_results_sent_twice = list(bert_results_sent_twice.values())
                if multi_task:
                    bert_results_both = json.load(open( os.path.join(res_root_dir,'rep_'+str(rep_i), 'finetune_result_train_ratio_{}_{}_num_labeled_{}'.format(t_r, 'pred_both', n) +'/finetune_final_results.json') ))
                    bert_results_both = list(bert_results_both.values())
                    bert_results_both = [[ll[0] for ll in l ] for l in bert_results_both ]
                for i,e_r in enumerate(eval_ratios):          
                    bert_best_results_sent = max(b[i] for b in bert_results_sent)
                    print ('bert_results_sent', bert_results_sent)
                    print ('bert_results_sent_twice', bert_results_sent_twice)
                    bert_best_results_sent_twice = max(b[i] for b in bert_results_sent_twice)

                    best_sent_accs[(t_r,e_r)][n]['bert_sent'] = bert_best_results_sent
                    best_sent_accs[(t_r,e_r)][n]['bert_sent_twice'] = bert_best_results_sent_twice
                    if multi_task:
                        bert_best_results_both = max([b[i] for b in bert_results_both])
                        best_sent_accs[(t_r,e_r)][n]['bert_both'] = bert_best_results_both

                    # bert_best_results_both = max(b[i] for b in bert_results_both)
                    # bert_best_results_both = [b[i] for b in bert_results_both][-1]
                    for strategy in strategies:
                        for finetune_type in ['pred_sent', 'pred_both']:
                            outdir = os.path.join(res_root_dir,'rep_'+str(rep_i),  'TripletBert_{}_{}_{}_num_labeled_{}'.format(t_r,strategy,finetune_type,n) )
                            results = json.load(open(outdir +'/transfer_results.json'))
                            results = [results[key] for key in results]
                            best_sent_accs[(t_r,e_r)][n][strategy +'_'+ finetune_type]= max([res[str(i)]['sent_enc'][0][0] for res in results])
        all_reps[rep_i] = best_sent_accs
    matplotlib.rcParams.update({'font.size': 25})
    final_res = {}
    err_bars = {}

    ##take average

    for t_r in train_ratios:
        for e_r in eval_ratios:
            final_res[(t_r,e_r)] = {}
            err_bars[(t_r,e_r)] = {}
            for n in num_train_datas:
                final_res[(t_r,e_r)][n] = {}
                err_bars[(t_r,e_r)][n] = {}
                keys = list(best_sent_accs[(t_r,e_r)][n].keys())
                for k in keys:
                    final_res[(t_r,e_r)][n][k] = []
                    err_bars[(t_r,e_r)][n][k] = []
                    for rep_i in rep_inds:
                        final_res[(t_r,e_r)][n][k].append( all_reps[rep_i][(t_r,e_r)][n][k])
                    
                    # max_val = max([ all_reps[rep_i][(t_r,e_r)][n][k] for rep_i in rep_inds ])
                    # min_val = min([ all_reps[rep_i][(t_r,e_r)][n][k] for rep_i in rep_inds ])
                    # final_res[(t_r,e_r)][n][k] /= len(rep_inds)
                    # err_bars[(t_r,e_r)][n][k] = [final_res[(t_r,e_r)][n][k] - min_val, max_val - final_res[(t_r,e_r)][n][k]]
                    # err_bars[(t_r,e_r)][n][k] = 2.776 * ( np.std([ all_reps[rep_i][(t_r,e_r)][n][k] for rep_i in rep_inds ]) / np.sqrt(5))
                    # breakpoint()

    figname = 'best_sent_acc_plot_each_rep.png'
    fig, axes = plt.subplots(nrows=len(train_ratios) * len(rep_inds), ncols=len(eval_ratios), figsize=(35,35), sharey=True)
    for rep_i, rep_num in enumerate(rep_inds):
        for i,t_r in enumerate(train_ratios):
            for j,e_r in enumerate(eval_ratios):
                ax = axes[rep_i+ i * len(train_ratios),j]
                # print (np.asarray([err_bars[(t_r,e_r)][n]['bert_sent']  for n in num_train_datas]).T)
                ax.plot([2 * n for n in num_train_datas], [final_res[(t_r,e_r)][n]['bert_sent'][rep_i]  for n in num_train_datas],'k:',\
                    label = 'bert_sent',linewidth=6.0)
                ax.plot([2 * n for n in num_train_datas], [final_res[(t_r,e_r)][n]['bert_sent_twice'][rep_i]   for n in num_train_datas],'g:',\
                    label = 'bert_sent_twice',linewidth=6.0)
                ax.plot([2 * n for n in num_train_datas], [final_res[(t_r,e_r)][n]['bert_both'][rep_i]  for n in num_train_datas],'b:', label = 'bert_both',linewidth=6.0)
                # ax.set_xlabel('Num_training_data')
                ax.set_xticks(num_train_datas)
                # ax.set_ylabel('Sentiment accuracy')
                ax.set_title('Train {}, eval {}, rep_num {}'.format(t_r,e_r, rep_num))
                colors = ['r','y','g', 'c']
                handles, labels = ax.get_legend_handles_labels()
                for k,strategy in enumerate(strategies):
                    c = colors[k]
                    ax.plot([2 * n for n in num_train_datas], [final_res[(t_r,e_r)][n][strategy +'_'+ 'pred_sent'][rep_i]   for n in num_train_datas],c+'-', label = strategy +'_'+ 'pred_sent', \
                        )
                    ax.plot([2 * n for n in num_train_datas], [final_res[(t_r,e_r)][n][strategy +'_'+ 'pred_both'][rep_i]   for n in num_train_datas],c+'-.', label = strategy +'_'+ 'pred_both', \
                        )
                    handles, labels = ax.get_legend_handles_labels()
            # ax.legend()

    fontP = FontProperties()
    fontP.set_size('small')
    fig.legend(handles, labels, loc='lower right', prop = fontP)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir,figname))

def collect_Bert_results(res_root_dir = '/dataxyz/experiments/DBert/1_28_1320' ,save_dir = None, train_ratios = [0.05,0.25,0.5],\
    eval_ratios = [0.05,0.25,0.5,0.75,0.95],num_train_datas = [25,50,100,200,400],rep_inds = [0,1,2,3,4], \
    multi_task = False):
    all_reps ={}
    for rep_i in rep_inds:
        best_sent_accs = {(t_r,e_r):{n:{}  for n in num_train_datas} for e_r in eval_ratios for t_r in train_ratios}
        for n in num_train_datas:
            for t_r in train_ratios:
                bert_results_sent= json.load(open( os.path.join(res_root_dir,'rep_'+str(rep_i), 'finetune_result_train_ratio_{}_{}_num_labeled_{}'.format(t_r, 'pred_sent', n) +'/finetune_final_results.json') ))
                bert_results_sent = list(bert_results_sent.values())
                bert_results_sent_twice= json.load(open( os.path.join(res_root_dir,'rep_'+str(rep_i), 'finetune_result_train_ratio_{}_{}_num_labeled_{}'.format(t_r, 'pred_sent', 2 * n) +'/finetune_final_results.json') ))
                bert_results_sent_twice = list(bert_results_sent_twice.values())
                if multi_task:
                    bert_results_both = json.load(open( os.path.join(res_root_dir,'rep_'+str(rep_i), 'finetune_result_train_ratio_{}_{}_num_labeled_{}'.format(t_r, 'pred_both', n) +'/finetune_final_results.json') ))
                    bert_results_both = list(bert_results_both.values())
                    bert_results_both = [[ll[0] for ll in l ] for l in bert_results_both ]
                for i,e_r in enumerate(eval_ratios):          
                    bert_best_results_sent = max(b[i] for b in bert_results_sent)
                    print ('bert_results_sent', bert_results_sent)
                    print ('bert_results_sent_twice', bert_results_sent_twice)
                    bert_best_results_sent_twice = max(b[i] for b in bert_results_sent_twice)

                    best_sent_accs[(t_r,e_r)][n]['bert_sent'] = bert_best_results_sent
                    best_sent_accs[(t_r,e_r)][n]['bert_sent_twice'] = bert_best_results_sent_twice
                    if multi_task:
                        bert_best_results_both = max([b[i] for b in bert_results_both])
                        best_sent_accs[(t_r,e_r)][n]['bert_both'] = bert_best_results_both
        all_reps[rep_i] = best_sent_accs
    final_res = {}
    err_bars = {}

    ##collect into a list

    for t_r in train_ratios:
        for e_r in eval_ratios:
            final_res[(t_r,e_r)] = {}
            err_bars[(t_r,e_r)] = {}
            for n in num_train_datas:
                final_res[(t_r,e_r)][n] = {}
                err_bars[(t_r,e_r)][n] = {}
                keys = list(best_sent_accs[(t_r,e_r)][n].keys())
                for k in keys:
                    final_res[(t_r,e_r)][n][k] = []
                    err_bars[(t_r,e_r)][n][k] = []
                    for rep_i in rep_inds:
                        final_res[(t_r,e_r)][n][k].append( all_reps[rep_i][(t_r,e_r)][n][k])
    return final_res

def collect_TripletBert_results(res_root_dir = None ,save_dir = None, train_ratios = [0.05,0.25,0.5],\
    eval_ratios = [0.05,0.25,0.5,0.75,0.95],num_train_datas = [25,50,100,200,400],rep_inds = [0,1], \
        strategies = ['only_diff', 'opposite', 'mixture','only_same'], show = True, multi_task = False):
    all_reps ={}
    for rep_i in rep_inds:
        best_sent_accs = {(t_r,e_r):{n:{}  for n in num_train_datas} for e_r in eval_ratios for t_r in train_ratios}
        for n in num_train_datas:
            for t_r in train_ratios:
                for i,e_r in enumerate(eval_ratios):       
                    for strategy in strategies:
                        for finetune_type in ['pred_sent', 'pred_both']:
                            outdir = os.path.join(res_root_dir,'rep_'+str(rep_i),  'TripletBert_{}_{}_{}_num_labeled_{}'.format(t_r,strategy,finetune_type,n) )
                            results = json.load(open(outdir +'/transfer_results.json'))
                            results = [results[key] for key in results]
                            best_sent_accs[(t_r,e_r)][n][strategy +'_'+ finetune_type]= max([res[str(i)]['sent_enc'][0][0] for res in results])   
                    
        all_reps[rep_i] = best_sent_accs
    final_res = {}
    ##collect into a list
    for t_r in train_ratios:
        for e_r in eval_ratios:
            final_res[(t_r,e_r)] = {}
            for n in num_train_datas:
                final_res[(t_r,e_r)][n] = {}
                keys = list(best_sent_accs[(t_r,e_r)][n].keys())
                for k in keys:
                    final_res[(t_r,e_r)][n][k] = []
                    for rep_i in rep_inds:
                        final_res[(t_r,e_r)][n][k].append( all_reps[rep_i][(t_r,e_r)][n][k])
    return final_res


def main():
    rep_ind = 1
    figname_fmt = "TripletBert_{}_{}_{}_num_labeled_{}_rep_{}.png"
    res_root_dir = '/home/xiongyi/dataxyz/experiments/DBert/1_29/rep_' +str(rep_ind) 
    
    matplotlib.rcParams.update({'font.size': 25})
    
    ERR = True
    # savedir = '/home/xiongyi/dataxyz/experiments/DBert/1_29_plots_ind_fixed'
    # if not os.path.exists(savedir):
        # os.makedirs(savedir)
    # for train_ratio in [0.05,]:
    #     for strategy in ['opposite']:
    #         for finetune_type in ['pred_sent', 'pred_both']:
    #             for num_labeled in [25, 50,100,200]:
    #                 figname = figname_fmt.format(train_ratio,strategy,finetune_type,num_labeled, rep_ind)
    #                 outdir = os.path.join(res_root_dir ,'TripletBert_{}_{}_{}_num_labeled_{}'.format(train_ratio,strategy,finetune_type,num_labeled) )
    #                 try:
    #                     train_losses = json.load(open(os.path.join(outdir, 'train_losses.json')))
    #                 except:
    #                     print (outdir , ' NOT FOUND ')
    #                     continue
    #                 val_losses = json.load(open(os.path.join(outdir, 'val_losses.json')))
    #                 transfer_results = json.load(open(os.path.join(outdir,'transfer_results.json')))
    #                 if not os.path.exists(savedir):
    #                     os.makedirs(savedir)
    #                 plot_results_DEBert( train_losses, val_losses, transfer_results,savedir, figname,model_type = 'TripletBert',eval_ratios = [0.05,0.25,0.5,0.75,0.95])
    
    final_res = collect_Bert_results(res_root_dir = '/home/xiongyi/dataxyz/experiments/DBert/1_28_1320' ,train_ratios = [0.05],\
    eval_ratios = [0.05,0.25,0.5,0.75,0.95],num_train_datas = [25,50,100],rep_inds = [0,1,2,3,4], \
    multi_task = False)

    final_triplet_res = collect_TripletBert_results(res_root_dir = '/home/xiongyi/dataxyz/experiments/DBert/1_29' , train_ratios = [0.05],\
    eval_ratios = [0.05,0.25,0.5,0.75,0.95],num_train_datas = [25,50,100,200],rep_inds = [1],strategies=['opposite'], multi_task = False)

    save_dir = '/home/xiongyi/dataxyz/experiments/DBert/compare_with_rep_bert/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    eval_ratios = [0.05,0.25,0.5,0.75,0.95]
    strategies = ['opposite']
    figname = 'best_sent_acc_multiple_bert.png'
    fig, axes = plt.subplots(nrows=1, ncols=len(eval_ratios), figsize=(35,35), sharey=True)
    t_r = 0.05
    num_train_datas = [25,50,100]
    for j,e_r in enumerate(eval_ratios):
        ax = axes[j]
        sent_res = []
        sent_twice_res = []
        for n in num_train_datas:
            all_sent_res = final_res[(t_r,e_r)][n]['bert_sent']
            all_sent_twice_res = final_res[(t_r,e_r)][n]['bert_sent_twice']
            sent_res.append([np.mean(all_sent_res), min(all_sent_res), max(all_sent_res)  ])
            sent_twice_res.append([np.mean(all_sent_twice_res), min(all_sent_twice_res), max(all_sent_twice_res)  ])
        ys = [s[0] for s in sent_res]
        yerr = [[s[0]-s[1], s[2]-s[0]] for s in sent_res]
        yerr = np.asanyarray(yerr).T

        ys_twice = [s[0] for s in sent_twice_res]
        yerr_twice = [[s[0]-s[1], s[2]-s[0]] for s in sent_twice_res]
        yerr_twice = np.asanyarray(yerr_twice).T
        # print (np.asarray([err_bars[(t_r,e_r)][n]['bert_sent']  for n in num_train_datas]).T)
        if ERR:
            ax.errorbar(x = [2 * n for n in num_train_datas], y = ys,yerr= yerr ,fmt = 'b:',\
            label = 'bert_sent',linewidth=6.0, elinewidth = 2.0,capsize = 15,capthick = 5)
            ax.errorbar(x = [2 * n for n in num_train_datas], y = ys_twice,yerr= yerr_twice ,fmt = 'g:',\
            label = 'bert_sent_twice',linewidth=6.0, elinewidth = 2.0,capsize = 15,capthick = 5)
        else:
            ax.plot([1.9 * n for n in num_train_datas], [np.mean(final_res[(t_r,e_r)][n]['bert_sent'])  for n in num_train_datas],'b:',\
                label = 'bert_sent',linewidth=6.0)
            ax.plot([2.1 * n for n in num_train_datas], [np.mean(final_res[(t_r,e_r)][n]['bert_sent_twice'])  for n in num_train_datas],'g:',\
                label = 'bert_sent_twice',linewidth=6.0)
        # ax.plot([2 * n for n in num_train_datas], [final_res[(t_r,e_r)][n]['bert_both']  for n in num_train_datas],'b:', label = 'bert_both',linewidth=6.0)
        # ax.set_xlabel('Num_training_data')
        # ax.set_ylabel('Sentiment accuracy')
        ax.set_title('Train {}, eval {}'.format(t_r,e_r))
        colors = ['r','y','g', 'c']
        handles, labels = ax.get_legend_handles_labels()
        for k,strategy in enumerate(strategies):
            c = colors[k]
            ax.plot([2 * n for n in num_train_datas], [np.mean(final_triplet_res[(t_r,e_r)][n][strategy +'_'+ 'pred_sent'])  for n in num_train_datas],c+'-', label = strategy +'_'+ 'pred_sent', \
                )
            ax.plot([2 * n for n in num_train_datas], [np.mean(final_triplet_res[(t_r,e_r)][n][strategy +'_'+ 'pred_both'])   for n in num_train_datas],c+'-.', label = strategy +'_'+ 'pred_both', \
                )
            handles, labels = ax.get_legend_handles_labels()
        # ax.legend()

    fig.legend(handles, labels, loc='lower right')
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir,figname))


    # compare_best_acc_across_reps(res_root_dir= '/home/xiongyi/dataxyz/experiments/DBert/1_29/' ,\
    #     save_dir =savedir, train_ratios = [0.05,0.05],rep_inds = [1],eval_ratios = [0.05,0.25,0.5,0.75,0.95],\
    #         num_train_datas = [25,50,100, 200,], strategies=['opposite'], multi_task = True)


if __name__ == '__main__':
    main()
    