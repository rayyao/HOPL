from HOPL.lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.prj_dir='/home/zl/code_of_HOPL/HOPL'
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.network_path = '/lib/test/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/lib/test/result_plots/'
    settings.results_path = '/lib/test/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/lib/test/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.save_dir='/home/zl/ThreePrompt/HOPL/results'

    return settings

# class pix_module(nn.Module):
#     '''
#     多特征融合 AFF
#     '''
#     def __init__(self, in_dim1=16, in_dim2=8,dropout=0.1):
#         super(pix_module, self).__init__()
#         self.proj_q1 = nn.Linear(in_dim1, in_dim1, bias=False)
#         self.proj_k1 = nn.Linear(in_dim1, in_dim1, bias=False)
#         self.proj_v1 = nn.Linear(in_dim1, in_dim1, bias=False)
#         self.proj_q2 = nn.Linear(in_dim2, in_dim2, bias=False)
#         self.proj_k2 = nn.Linear(in_dim2, in_dim2, bias=False)
#         self.proj_v2 = nn.Linear(in_dim2, in_dim2, bias=False)
#         self.dropout1 = nn.Dropout(dropout)
#     def forward(self, x1, x2):
#         a,b,c,d=x1.size()
#         if c==16:
#             q = self.proj_q1(x2)
#             k = self.proj_k1(x1)
#             v = self.proj_v1(x1)
#         else:
#             q = self.proj_q2(x2)
#             k = self.proj_k2(x1)
#             v = self.proj_v2(x1)
#         out=torch.matmul(q,k.transpose(-2,-1))/torch.sqrt(torch.tensor(k.size(-1)).float())
#         out=torch.tensor(out)
#         out=out.to(torch.float32)
#         atten_weights=F.softmax(out,dim=-1)
#         atten_out=torch.matmul(atten_weights,v)
#         x = x1 + self.dropout1(atten_out)
#         return x