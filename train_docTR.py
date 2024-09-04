import os
import time
import cv2
import gc
import json
import shutil
from numba import cuda
import numpy as np
import tensorflow as tf
from datasets import ClassLabel, Sequence, Value, Features, Array2D, Array3D
from torch.utils.data import DataLoader
import tqdm
from datetime import datetime
import torch
from PIL import Image,ImageDraw,ImageFont

from datasets import ClassLabel, Sequence, Value, Features, Array2D, Array3D
from transformers import AutoProcessor ,AutoModelForTokenClassification, AdamW, get_scheduler

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score)

from autoai_process import Config
import pandas as pd

from datasets import Dataset

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from transformers import AutoConfig

def clear_memory(cuda_index=0):
    cuda.select_device(cuda_index) 
    cuda.close()

class RlefToLayoutLMDataConverter:
    def __init__(self):
        pass

    def bbox_method(self, json_file):
        data = json.loads(json_file)

        # Initialize lists to store classes and bounding box coordinates
        classes = []
        bounding_boxes = []

        # Iterate over each object in JSON data
        for item in data:
            # Append class to classes list
            classes.append(item["selectedOptions"][1]["value"])

            # Extract vertices to compute bounding box coordinates
            vertices = item["vertices"]
            x_coords = [vertex["x"] for vertex in vertices]
            y_coords = [vertex["y"] for vertex in vertices]

            # Compute bounding box coordinates
            min_x = int(min(x_coords))
            max_x = int(max(x_coords))
            min_y = int(min(y_coords))
            max_y = int(max(y_coords))

            # Append bounding box coordinates to bounding_boxes list
            bounding_boxes.append([min_x, min_y, max_x, max_y])
            
        return classes, bounding_boxes
    
    def resize_image_and_bboxes(self, image, bboxes, max_size=1000):
        original_height, original_width = image.shape[:2]

        scale_factor = min(max_size / original_width, max_size / original_height)

        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        resized_image = cv2.resize(image, (new_width, new_height))
        resized_bboxes = []
        
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            x_min = int(x_min * scale_factor)
            y_min = int(y_min * scale_factor)
            x_max = int(x_max * scale_factor)
            y_max = int(y_max * scale_factor)
            resized_bboxes.append([x_min, y_min, x_max, y_max])

        return resized_image, resized_bboxes

    # Resize images and their corresponding bounding boxes
    def resize_image_and_bboxes2(self, image_path, bboxes):
        bboxes2 = []

        # for i in range(len(paths)):
        # image_path = paths[i]
        image = cv2.imread(image_path)
        sample_bboxes = bboxes
        resized_image, resized_bboxes1 = self.resize_image_and_bboxes(image.copy(), sample_bboxes, max_size=1000) 
        bboxes2.append(resized_bboxes1)
        cv2.imwrite(image_path, resized_image)
        return bboxes2
    
    def get_intersection_percent(self, big_box, small_box):
        x_left = max(big_box[0], small_box[0])
        y_top = max(big_box[1], small_box[1])
        x_right = min(big_box[2], small_box[2])
        y_bottom = min(big_box[3], small_box[3])

        # no intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # area of the smaller bbox
        small_box_area = (small_box[2] - small_box[0]) * (small_box[3] - small_box[1])
        
        if small_box_area == 0:
            return 0.0

        # Calculate percentage
        return (intersection_area / small_box_area) * 100
    
    def docTR_results(self, extracted_text):
        words = []
        boxes = []
        conf = []

        height, width = extracted_text['pages'][0]['dimensions']
        for page in extracted_text['pages']:
            for block in page['blocks']:
                for line in block['lines']:
                    for word in line['words']:
                        text = word['value']
                        (xmin_norm, ymin_norm), (xmax_norm, ymax_norm) = word['geometry']

                        xmin = int(xmin_norm * width)
                        ymin = int(ymin_norm * height)
                        xmax = int(xmax_norm * width)
                        ymax = int(ymax_norm * height)

                        bbox = [xmin, ymin, xmax, ymax]

                        score = word['objectness_score']
                        # print(text, bbox, score * 100, end=' ')

                        words.append(text)
                        boxes.append(bbox)
                        conf.append(score)
        return words, boxes, conf
    
    def find_boxes_within(self, big_boxes, small_boxes, labels, threshold = 70):
        final_labels = []
        for i, small_box in enumerate(small_boxes):
            found = False
            for j, big_box in enumerate(big_boxes):
                percent_inside = self.get_intersection_percent(big_box, small_box)
                if percent_inside > threshold:
                    final_labels.append(labels[j])
                    found = True
                    break
            if not found:
                final_labels.append("others")
        return final_labels

    # Generate dataset from annotations, labels, and image paths
    def make_dataset(self, dataset_csv, dataset_path, docTR_predictor = None, file_save = "train.csv", eval_set = False):
        image_paths = []
        rlef_labels = []
        rlef_bboxes = []
        
        ocr_words = []
        ocr_boxes = []
        
        print("="*100)
        print("Data Prep Started")
        print("="*100)
        
        # print("="*100)
        # print("Initializing model")
        # docTR_predictor = ocr_predictor(pretrained=True, assume_straight_pages=True)
        # print("Model Initialized")
        # print("="*100)
        
        dataset_csv = pd.read_csv(dataset_csv)
        
        if eval_set:
            org_image_paths = []
            os.makedirs(os.path.join(dataset_path, "org"), exist_ok=True)
    
        for i in range(len(dataset_csv)):
            annotations = dataset_csv["imageAnnotations"][i]
            resource_ids = dataset_csv['_id'][i]
            gcs_filename = dataset_csv['GCStorage_file_path'][i]
            label = dataset_csv['label'][i]
            
            image_path = os.path.join(dataset_path, "images", os.path.basename(gcs_filename))
            image_paths.append(image_path)
            
            if eval_set:
                org_image_path = os.path.join(dataset_path, "org", os.path.basename(gcs_filename))
                org_image_paths.append(org_image_path)
                shutil.copy(image_path, org_image_path)
            
            print("="*100)
            print(f"Inference on image: #{i + 1} started")
            print(image_path)
            
            start = time.time()
            doc = DocumentFile.from_images([image_path])
            result = docTR_predictor(doc)
            end = time.time()
            
            print(f"Inference on image complete in {end - start} seconds")
            print("="*100)
            
            o_words, o_boxes, o_conf = self.docTR_results(result.export())

            r_labels, r_bboxes = self.bbox_method(json_file=annotations)
            
            rlef_labels.append(r_labels)
            rlef_bboxes.append(r_bboxes)
            
            ocr_words.append(o_words)
            ocr_boxes.append(o_boxes)
            
        all_labels = Config.all_labels
        all_labels.sort()
        
        id2label = {v: k for v, k in enumerate(all_labels)}
        label2id = {k: v for v, k in enumerate(all_labels)}
        
        total_labels = []
        res_bboxes = []
        
        for i in range(len(rlef_bboxes)): 
            im_labels = self.find_boxes_within(big_boxes = rlef_bboxes[i], 
                                         small_boxes = ocr_boxes[i], 
                                         labels = rlef_labels[i], 
                                        threshold = 70)
            total_labels.append(im_labels)
            
            res_box = self.resize_image_and_bboxes2(image_paths[i], ocr_boxes[i])
            res_bboxes.append(res_box[0])
        
        ner_labels = []
        final_bboxes = []
        final_words = []
        for i, labelx in enumerate(total_labels):
            temp_word = []
            temp_ner_label = []
            temp_bboxes = []
            for j, an in enumerate(labelx):
                try:
                    temp_ner_label.append(label2id[an])
                    temp_word.append(ocr_words[i][j])
                    temp_bboxes.append(res_bboxes[i][j])
                except:
                    pass
            ner_labels.append(temp_ner_label)
            final_bboxes.append(temp_bboxes)
            final_words.append(temp_word)
        
        data_df = pd.DataFrame()
        if eval_set:
            data_df["org_image_path"] = org_image_paths
        data_df["bboxes"] = final_bboxes
        data_df["ner_tags"] = ner_labels
        data_df["image_path"] = image_paths
        data_df["words"] = final_words
    
        data_df.to_csv(file_save, index=False)
        print("="*100)
        print("Data Preperation finished")
        print("="*100)
        
        # data_df = pd.DataFrame()
        # all_labels = []
        # datasets = Dataset.from_pandas(data_df)

        return data_df
    
class PrepareEncodings:
    def __init__(self, data, processor):
        self.processor = processor
        self.df = data

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        data_point = self.df.iloc[idx]

        image = Image.open(data_point["image_path"]).convert("RGB")
        # words = data_point["words"]
        # labels = data_point["ner_tags"]
        # bboxes = data_point["bboxes"]

        words = eval(str(data_point["words"]))
        labels = eval(str(data_point["ner_tags"]))
        bboxes = eval(str(data_point["bboxes"]))

        encoding = self.processor(image, words, word_labels=labels, boxes=bboxes,
                                  padding="max_length", truncation=True, return_tensors="pt")

        for k,v in encoding.items():
            encoding[k] = v.squeeze()
        return encoding

class LayoutLMTrainer:
    def __init__(self, processor_revision="no_ocr", all_labels = []):
        print("LayoutLMTrainer Initialization...\n\n\n\n")
        
        self.all_labels = all_labels
        self.id2label = {v: k for v, k in enumerate(self.all_labels)}
        self.label2id = {k: v for v, k in enumerate(self.all_labels)}
            
        print(">> All_labels and ids\n", self.all_labels)
        print(self.id2label)
        print(self.label2id)
        
        self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        self.model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=len(all_labels), id2label = self.id2label, label2id = self.label2id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(" >> Self Device: ", self.device)
        self.model.to(self.device)
        print("="*80)

    def data_preparation(self, dataset, batch_size):
        dataset = PrepareEncodings(dataset, self.processor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(">> dataloader created")
        print(dataloader)
        return dataloader
    
    def get_scheduler(self, optimizer, num_warmup_steps, num_training_steps):
        
        return get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def train_fn(self, train_dataloader, optimizer):
        train_losses = []
        self.model.train()
        train_loss = 0.0
        step = 0

        pbar = tqdm.tqdm(train_dataloader, desc="Training", total = len(train_dataloader))
        for batch in pbar:
            # get the inputs;
            input_ids = batch['input_ids'].to(self.device)
            bbox = batch['bbox'].to(self.device)
            pixel_values = batch['pixel_values'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(input_ids=input_ids,
                            bbox=bbox,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask,
                            labels=labels)

            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            step += 1

        train_loss = train_loss / step
        train_losses += [train_loss]
        pbar.set_postfix({"train_loss": train_loss})
        return optimizer, train_losses
    
    def result_score(self, predictions, ground_truths, labels, report= False, model_dir = None, epoch = None):
        predictions = np.argmax(predictions, axis=2)
        label_map = {i: label for i, label in enumerate(labels)}
        indices = np.where(ground_truths == -100)
        preds_list = []
        gts_list = []

        model_name = "new_model"

        for i in range(ground_truths.shape[0]):
            pred_copy = predictions[i].copy()
            gt_copy = ground_truths[i].copy()
            pred_copy = np.delete(pred_copy, indices[1][indices[0] == i])
            gt_copy = np.delete(gt_copy, indices[1][indices[0] == i])
            preds_list.append([label_map[pred] for pred in pred_copy])
            gts_list.append([label_map[label] for label in gt_copy])

        print("Evaluation completed\n")
        if report == True and model_dir and epoch != None:
            _report = classification_report(gts_list, preds_list)
            print(_report)
            with open(model_dir + f"/report_{model_name}.txt", "a") as file_object:
                file_object.write(f"{epoch} number\n"+ _report + "\n\n")

        results = {
        "precision": precision_score(gts_list, preds_list),
        "recall": recall_score(gts_list, preds_list),
        "f1": f1_score(gts_list, preds_list),
        }

        return results

    def eval_fn(self, test_dataloader, all_labels, report= False, model_dir = None, epoch = None):
        eval_losses = []
        eval_loss = 0.0
        step = 0

        pred_vals = None
        ground_truths = None

        self.model.eval()
        pbar = tqdm.tqdm(test_dataloader, desc="Evaluating", total = len(test_dataloader))
        for batch in pbar:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(self.device)
                bbox = batch['bbox'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # forward pass
                outputs = self.model(input_ids=input_ids, bbox=bbox, pixel_values=pixel_values, attention_mask=attention_mask,
                                labels=labels)

                loss = outputs.loss
                eval_loss += loss.item()
                step += 1

                if pred_vals is None:
                    pred_vals = outputs.logits.detach().cpu().numpy()
                    ground_truths = batch["labels"].detach().cpu().numpy()
                else:
                    pred_vals = np.append(pred_vals, outputs.logits.detach().cpu().numpy(), axis=0)
                    ground_truths = np.append(
                        ground_truths, batch["labels"].detach().cpu().numpy(), axis=0
                    )

        eval_loss = eval_loss / step
        eval_losses += [eval_loss]

        result = self.result_score(pred_vals, ground_truths, all_labels, report= report, model_dir = model_dir, epoch = epoch)
        result["test_loss"] = eval_loss
        return result
    
    def write_json(self, new_data, filename= ""):
        with open(filename,'r+') as file:
            file_data = json.load(file)
            file_data["epoch_details"].append(new_data)
            file.seek(0)
            json.dump(file_data, file, indent = 4)

    def train(self, train_dataset, eval_dataset, epochs, batch_size, model_saving_path):
        train_dataloader = self.data_preparation(train_dataset, batch_size)
        eval_dataloader = self.data_preparation(eval_dataset, batch_size)

        model_name = "new_model"

        with open(model_saving_path + f"/report_{model_name}.txt", "w") as file_object:
            file_object.write("")

        filename= model_saving_path +f"/run_results_{model_name}.json"

        with open(filename,'w') as file:
                file_data = {}
                file_data["epoch_details"] = []
                json.dump(file_data, file, indent = 4)

        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        
        num_training_steps = epochs * len(train_dataloader)
        num_warmup_steps = num_training_steps // 10

        # Create the learning rate scheduler
        lr_scheduler = self.get_scheduler(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)
        

        MODEL_PATH = model_saving_path + f"/model_{model_name}.bin"
        best_f1_score = -1
        best_epoch = 1
        for epoch in range(epochs):
            optimizer, train_losses = self.train_fn(train_dataloader, optimizer)
            current_f1_score = self.eval_fn(eval_dataloader, self.all_labels, report = True, model_dir = model_saving_path, epoch = epoch)
            print(current_f1_score)
            self.write_json(current_f1_score, filename= model_saving_path +f"/run_results_{model_name}.json")
            if current_f1_score["f1"] > best_f1_score:
                
                torch.save(self.model.state_dict(), model_saving_path + f"/pytorch_model.bin")
                torch.save(optimizer.state_dict(), model_saving_path + f"/optimizer.pt")

                # Save the fine-tuned model's config
                self.model.config.save_pretrained(model_saving_path)
                
                best_f1_score = current_f1_score["f1"]
                best_epoch = epoch
            print(f"best_f1_score : {best_f1_score} at epoch number: {best_epoch + 1}" , f"--> Current epoch: {epoch + 1}")
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs} - Learning rate: {current_lr:.2e}")
            
            lr_scheduler.step()


def main(train_dataset_path, eval_dataset_path, models_path, model_details, train_dataset_csv_path, eval_dataset_csv_path):
    trained_model_path = os.path.join(models_path, f'llmv3_{datetime.now().strftime("%Y-%m-%d_%H-%M")}')
    os.makedirs(trained_model_path, exist_ok=True)

#     train_dataset_csv = pd.read_csv(train_dataset_csv_path)
#     eval_dataset_csv = pd.read_csv(eval_dataset_csv_path)

    print("="*100)
    print("Initializing model")
    docTR_predictor = ocr_predictor(pretrained=True).to(torch.device('cpu'))
    print("Model Initialized")
    print("="*100)
    
    all_labels = Config.all_labels
    all_labels.sort()
    
    print(all_labels)
    
    preprocessor = RlefToLayoutLMDataConverter()
    
    train_dataset = preprocessor.make_dataset(train_dataset_csv_path, train_dataset_path, docTR_predictor = docTR_predictor, file_save = "train.csv")
    eval_dataset = preprocessor.make_dataset(eval_dataset_csv_path, eval_dataset_path, docTR_predictor = docTR_predictor, file_save = "eval.csv", eval_set = True)

    model = LayoutLMTrainer(all_labels = all_labels)
    model.train(train_dataset, eval_dataset, epochs = model_details["hyperParameter"]["epochs"], batch_size = model_details["hyperParameter"]["batch"], model_saving_path = trained_model_path)
    
    return trained_model_path
