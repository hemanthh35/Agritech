# Run this ONCE in Colab after train_data = train_gen.flow_from_directory(...)
# to export class label order that matches your saved Keras model output indices.
#
# import json
# ordered = [None] * len(train_data.class_indices)
# for name, idx in train_data.class_indices.items():
#     ordered[idx] = name
# with open("class_labels.json", "w", encoding="utf-8") as f:
#     json.dump(ordered, f, indent=2)
# files.download("class_labels.json")
#
# Copy the downloaded file to agritech/models/class_labels.json (replace default).
