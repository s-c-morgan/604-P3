import scripts.plot_eda 
import scripts.plot_group_series
import scripts.test_bag_effect_for_infridge
import scripts.test_fridge_layer_effects 

scripts.plot_eda.plot_eda_heatmaps(features=['hue', 'saturation', 'value'], save_dir='plots')
scripts.plot_eda.plot_eda_catplot(features=['value'], save_dir='plots')
scripts.plot_group_series.generate_plots_img_feature_by_group(criteria=['green'], window=3, save_dir='plots')
scripts.plot_group_series.generate_plot_weight_by_group(window=3, save_dir='plots')
scripts.test_fridge_layer_effects.layer_difference_test_img_feature(criteria=['value'], save_dir='plots')
scripts.test_fridge_layer_effects.layer_difference_test_weight(save_dir='plots')
scripts.test_bag_effect_for_infridge.bagvsnot_test_img_feature(criteria = 'green',save_dir='plots')
scripts.test_bag_effect_for_infridge.bagvsnot_test_weight(save_dir='plots')