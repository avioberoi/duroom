"""Pass of a public-domain artwork through CLIP: token norms per block, the end vector and its angle to the corpus mean.
Usage: python image_to_vector.py"""
from _panels import *
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Circle, Arc, FancyArrowPatch, ConnectionPatch

d=load_data('image_to_vector')
crop=plt.imread(HERE/d['crop_asset'])
cmap=LinearSegmentedColormap.from_list('token-length',[PAPER,ACCENT])

def token_map(ax, block):
    ax.imshow(block['norms'],cmap=cmap,norm=LogNorm(block['min'],block['max']),interpolation='nearest')
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)

def scale_bar(fig,x,y,w,block,mobile):
    ax=fig.add_axes([x,y,w,0.008 if mobile else 0.012])
    ax.imshow(np.linspace(0,1,128)[None,:],aspect='auto',cmap=cmap)
    ax.axis('off')
    for v,pos,align in [(block['min'],x,'left'),(block['max'],x+w,'right')]:
        fig.text(pos,y-0.008,f'{v:.1f}',ha=align,va='top',fontsize=9,color=MUTE)

for mobile in (False,True):
    fig=plt.figure(figsize=(3.9,12.8) if mobile else (8.8,7.7))
    fig.text(0.5,0.985,'A measured pass through CLIP',ha='center',va='top',fontsize=14)
    if mobile:
        input_box=[0.30,0.805,0.40,0.135]
        boxes=[[x,0.642,0.24,0.081] for x in (0.065,0.38,0.695)]
        title_y,bar_y,class_y=0.733,0.625,0.586
    else:
        input_box=[0.04,0.65,0.20,0.23]
        boxes=[[x,0.65,0.20,0.23] for x in (0.29,0.53,0.77)]
        title_y,bar_y,class_y=0.90,0.625,0.572
    a=fig.add_axes(input_box)
    a.imshow(crop)
    grid=d['patch_grid'];size=d['crop_size'][0]
    for edge in np.linspace(-0.5,size-0.5,grid+1):
        a.axhline(edge,color=PAPER,lw=0.35,alpha=0.65);a.axvline(edge,color=PAPER,lw=0.35,alpha=0.65)
    a.set_xlim(-.5,size-.5);a.set_ylim(size-.5,-.5);a.axis('off')
    fig.text(input_box[0]+input_box[2]/2,0.954 if mobile else title_y,'The input, in patches',ha='center',fontsize=12)
    fig.text(input_box[0]+input_box[2]/2,0.78 if mobile else 0.617,
             f'{size} \u00d7 {size} crop\n{grid} \u00d7 {grid} patches; sides removed',ha='center',va='top',fontsize=10,color=MUTE)
    maps=[]
    for box,block in zip(boxes,d['blocks']):
        ax=fig.add_axes(box);token_map(ax,block);maps.append(ax)
        fig.text(box[0]+box[2]/2,title_y,f'After block {block["block"]}',ha='center',fontsize=11)
        scale_bar(fig,box[0],bar_y,box[2],block,mobile)
    if not mobile:
        for left,right in zip([a]+maps, maps):
            fig.add_artist(ConnectionPatch((1.05,.5),(-.05,.5),coordsA='axes fraction',coordsB='axes fraction',
                axesA=left,axesB=right,arrowstyle='-|>',mutation_scale=9,color=MUTE,lw=.8,clip_on=False))
    note=(f'Each patch becomes {d["token_dimensions"]:,} numbers.\n'
          'Colour shows length, not importance.\n'
          'Very long tokens can serve internal computations.\n'
          'Separate logarithmic scales per block;\n'
          'not attention or decoded images.' if mobile else
          f'Each patch becomes {d["token_dimensions"]:,} numbers; maps show vector lengths, not attention or decoded images.\n'
          'Bright tiles do not mark important content: background tokens can serve internal computations.\n'
          'Colour scales are logarithmic and separate for each block.')
    fig.text(.5,0.593 if mobile else .557,note,ha='center',va='top',fontsize=10,color=MUTE,linespacing=1.4)

    # The end vector follows the final normalization and a learned linear map; its length is not on the 1024-wide block-norm scale above.
    v=fig.add_axes([.17,.343,.77,.10] if mobile else [.075,.185,.48,.19])
    raw=np.array(d['raw_values_rounded']);index=np.arange(1,len(raw)+1)
    v.vlines(index,0,raw,color=np.where(raw>=0,ACCENT,MUTE),lw=.65)
    v.axhline(0,color=HAIR,lw=.6);v.set_xlim(1,len(raw));v.set_xticks([1,len(raw)//2,len(raw)])
    clean_axes(v,mobile);v.set_xlabel('Coordinate',fontsize=10);v.set_ylabel('Value',fontsize=10)
    fig.text(.5 if mobile else .075,0.474 if mobile else .436,
             f'The image vector: {d["dimension"]} coordinates',ha='center' if mobile else 'left',fontsize=12)
    fig.text(.5 if mobile else .075,0.277 if mobile else .105,
             f'End-vector length: {d["length"]:.3f}',ha='center' if mobile else 'left',fontsize=11)
    fig.text(.5 if mobile else .075,0.255 if mobile else .075,
             'Coordinates are model values, not named styles.',ha='center' if mobile else 'left',fontsize=9,color=MUTE)

    e=fig.add_axes([.23,.055,.54,.15] if mobile else [.66,.115,.30,.29])
    e.set_xlim(-1.12,1.17);e.set_ylim(-1.1,1.18);e.set_aspect('equal');e.axis('off')
    e.add_patch(Circle((0,0),1,fill=False,edgecolor=HAIR,lw=1))
    point=np.array(d['sphere_point']);mean=d['corpus_mean_length']
    e.plot([0,1],[0,0],color=HAIR,lw=.8)
    e.add_patch(FancyArrowPatch((0,0),point,arrowstyle='-|>',mutation_scale=11,color=ACCENT,lw=1.2))
    e.scatter(*point,s=24,color=ACCENT,zorder=4)
    e.scatter(mean,0,s=20,color=MUTE,zorder=4);e.scatter(0,0,s=8,color=MUTE)
    e.add_patch(Arc((0,0),.7,.7,theta1=0,theta2=d['angle_deg'],color=MUTE,lw=.8))
    e.text(.44,.14,f'{d["angle_deg"]:.1f}\u00b0',fontsize=10,color=ACCENT)
    e.text(.33,-.22,'Corpus average',fontsize=9,ha='center',color=MUTE)
    e.text(0,1.13,'One point of length 1',fontsize=11,ha='center')
    if mobile:
        fig.text(.5,.226,f'Divide every coordinate by {d["length"]:.3f}',ha='center',fontsize=11,color=ACCENT)
    else:
        fig.text(.81,.436,f'Divide by {d["length"]:.3f}',ha='center',fontsize=12,color=ACCENT)
    example=d['sample_indices'][0]
    formula=f'For example: {raw[example]:+.5f} \u00f7 {d["length"]:.3f} = {d["unit_values"][example]:+.6f}'
    fig.text(.5,0.025 if mobile else .04,formula,ha='center',fontsize=10,color=MUTE)
    fig.text(.5,.007 if mobile else .014,
             'Circle: the plane of this work and the corpus average.',ha='center',fontsize=9,color=MUTE)
    finish(fig,'image_to_vector',mobile)
