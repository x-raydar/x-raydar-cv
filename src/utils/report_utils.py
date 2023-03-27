import numpy as np

def build_report(probs_multi, pred_urg):
    report = dict()

    report['AI_urgency'] =  pred_urg
    report['AI_prediction'] = build_prediction_report_multi(probs_multi)

    return report


def build_prediction_report_multi(probs):    
    num_labels = probs.size
    probs = probs.squeeze().tolist()
    prediction = dict(zip(load_list_radiologicalfindings(num_labels=num_labels), probs))

    return prediction


def get_priorityint_from_multi_probs(probs):

    num_labels = probs.size
    
    # From probs to list of positive findings
    list_positive_findings = list(np.array(load_list_radiologicalfindings(num_labels=num_labels))[probs > 0.5])

    # From list of positive findings to priority levels, and the to priosity ints
    list_priorityints = [dict_prioritylevel_priorityint[dict_finding_prioritylevel[f]] for f in list_positive_findings] + [0] # + Normal

    # Return max of the priority ints (max priority)
    return max(list_priorityints)

##########################################################################################
# Equivalent to labels_utils.load_list_radiologicalfindings(num_labels=43)
def load_list_radiologicalfindings(num_labels):
    if num_labels == 43:
        list_radiologicalfindings = ['abnormal_non_clinically_important',   # 0
                                    'aortic_calcification',
                                    'apical_fibrosis',
                                    'atelectasis',
                                    'axillary_abnormality',
                                    'bronchial_wall_thickening',           # 5
                                    'bulla',
                                    'cardiomegaly',
                                    'cavitating_lung_lesion',
                                    'clavicle_fracture',
                                    'consolidation',                       # 10
                                    'coronary_calcification',
                                    'dextrocardia',
                                    'dilated_bowel',
                                    'emphysema',
                                    'ground_glass_opacification',          # 15
                                    'hemidiaphragm_elevated',
                                    'hernia',
                                    'hyperexpanded_lungs',
                                    'interstitial_shadowing',
                                    'left_lower_lobe_collapse',            # 20
                                    'left_upper_lobe_collapse',
                                    'mediastinum_displaced',
                                    'mediastinum_widened',
                                    'object',
                                    'paraspinal_mass',                     # 25
                                    'paratracheal_hilar_enlargement',
                                    'parenchymal_lesion',
                                    'pleural_abnormality',
                                    'pleural_effusion',
                                    'pneumomediastinum',                   # 30
                                    'pneumoperitoneum',
                                    'pneumothorax',
                                    'rib_fracture',
                                    'rib_lesion',
                                    'right_lower_lobe_collapse',           # 35
                                    'right_middle_lobe_collapse',
                                    'right_upper_lobe_collapse',
                                    'scoliosis',
                                    'subcutaneous_emphysema',
                                    'unfolded_aorta',                      # 40
                                    'upper_lobe_blood_diversion',
                                    'volume_loss']
    elif num_labels == 38:
        list_radiologicalfindings = ['abnormal_non_clinically_important',   # 0
                                    'aortic_calcification',
                                    'apical_changes',
                                    'atelectasis',
                                    'axillary_abnormality',
                                    'bronchial_changes',            # 5
                                    'bulla',
                                    'cardiomegaly',
                                    'cavity',
                                    'clavicle_fracture',
                                    'consolidation',                        # 10
                                    'cardiac_calcification',
                                    'dextrocardia',
                                    'dilated_bowel',
                                    'emphysema',
                                    'ground_glass_opacification',           # 15
                                    'hemidiaphragm_elevated',
                                    'hernia',
                                    'hyperexpanded_lungs',
                                    'interstitial_shadowing',
                                    'mediastinum_displaced',                # 20
                                    'mediastinum_widened',
                                    'object',
                                    'paraspinal_mass',
                                    'paratracheal_hilar_enlargement',
                                    'parenchymal_lesion',                   # 25
                                    'pleural_abnormality',
                                    'pleural_effusion',
                                    'pneumomediastinum',
                                    'pneumoperitoneum',
                                    'pneumothorax',                         # 30
                                    'rib_fracture',
                                    'rib_lesion',
                                    'scoliosis',
                                    'subcutaneous_emphysema',
                                    'tortuosity_aorta',                       # 35
                                    'Pulmonary_bloodflow_redis.',
                                    'volume_loss']
        
    
    return list_radiologicalfindings

