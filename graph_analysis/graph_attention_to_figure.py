import matplotlib.pyplot as plt
import random
global_ = {'Shelf':0.44,'Cabinet':0.06,'CounterTop':0.12}
current_ = {'Cabinet':0.12,'CounterTop':0.22}
priori_ = {'Mug':0.02,'Cup':0.02,'WineBottle':0.02,'Laptop':0.02,'Bowl':0.02,'Box':0.02,'Kettle':0.01,'Pot':0.01,'Fridge':0.01,'Bed':0.01}

def method1():
    # https://stackoverflow.com/questions/28418988/how-to-make-a-histogram-from-a-list-of-strings-in-python
    for data in [global_, current_, priori_]:
        object_name, attention_score = data.keys(), data.values()
        ticks = range(len(object_name))
        plt.figure(figsize=(2, 3))
        plt.bar(ticks, attention_score, align='center', width=0.2)
        plt.xticks(ticks, object_name)
def method2():
    count = 0
    all_ticks = []
    all_object_name = []
    def plot_bar(data, ax, label):
        nonlocal count, all_ticks, all_object_name
        object_name, attention_score = data.keys(), data.values()
        ticks = range(count, count + len(object_name))
        count += len(object_name)
        ax.bar(ticks, attention_score, align='center', width=0.4, label=label)
        all_ticks.extend(ticks)
        all_object_name.extend(object_name)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig, ax = plt.subplots(1, 1)
    plot_bar(global_, ax, label="global_graph")
    plot_bar(current_, ax, label="current_graph")
    plot_bar(priori_, ax, label="priori_graph")
    plt.legend(["global_graph", "current_graph", "priori_graph"])
    # ax.set_xticks(all_ticks, minor=False)
    # ax.set_xticklabels(all_object_name, fontdict=None, minor=False)
    # ax.set_ylim([0, 1])
    # remove x tick # https://stackoverflow.com/questions/12998430/remove-xticks-in-a-matplotlib-plot
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    # https://www.geeksforgeeks.org/adding-labels-to-histogram-bars-in-matplotlib/
    # rects = ax.patches
    # for rect, label in zip(rects, all_object_name):
    #     height = rect.get_height()
    #     ran_n = 0.01 + random.uniform(0, 0.1)
    #     if height + ran_n > 1:
    #         ran_n = 0.01
    #     print(ran_n)
    #     ax.text(rect.get_x() + rect.get_width() / 2, height+ran_n, label,
    #             ha='center', va='bottom')

    # https://stackoverflow.com/questions/19073683/matplotlib-overlapping-annotations-text
    from adjustText import adjust_text
    texts = []
    rects = ax.patches
    for rect, label in zip(rects, all_object_name):
        height = rect.get_height()
        texts.append(
            plt.text(rect.get_x() + rect.get_width(), height + 0.02, label))
    adjust_text(texts, only_move={'points':'y', 'texts':'y'}, autoalign='y')#, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

method2()
plt.show()