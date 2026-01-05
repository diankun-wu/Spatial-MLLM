import json

def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def write_jsonl(data, file_path):
    """Writes a list of dictionaries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def process_vsi_590k(input_file, output_file):
    """Processes the VSI-590K dataset from JSONL format."""
    data = read_jsonl(input_file)
    print(f"Total records read: {len(data)}")
    if data:
        print(f"Sample record: {data[0]}")

    replacements = 0
    video_updates = 0
    image_updates = 0
    for idx, record in enumerate(data):
        conversation = record.get('conversations') or record.get('conversation')
        video_path = record.get('video')
        image_path = record.get('image')

        if isinstance(conversation, list):
            for message in conversation:
                if message.get('from') != 'human':
                    continue
                value = message.get('value')
                if isinstance(value, str) and '<image>' in value and isinstance(video_path, str):
                    message['value'] = value.replace('<image>', '<video>')
                    replacements += 1

        if isinstance(video_path, str) and not video_path.startswith('vsi-590k/'):
            record['video'] = f"vsi-590k/{video_path.lstrip('/')}"
            video_updates += 1
        elif video_path is not None and not isinstance(video_path, str):
            print(f"Unexpected video path format at record {idx}: {video_path}")

        if isinstance(image_path, str) and not image_path.startswith('vsi-590k/'):
            record['image'] = f"vsi-590k/{image_path.lstrip('/')}"
            image_updates += 1
        elif image_path is not None and not isinstance(image_path, str):
            print(f"Unexpected image path format at record {idx}: {image_path}")

    print(f"Total replacements: {replacements}")
    print(f"Total video updates: {video_updates}")
    print(f"Total image updates: {image_updates}")
    write_jsonl(data, output_file)
    print(f"Processed data written to: {output_file}")

if __name__ == "__main__":
    input_file = 'datasets/annotations/vsi_590k.jsonl'
    output_file = 'datasets/annotations/vsi-590k-processed.jsonl'
    process_vsi_590k(input_file, output_file)
