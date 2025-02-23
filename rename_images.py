import os
folder_path='raw_images/'
if not os.path.exists(folder_path):
        print(f"Error: The Folder '{folder_path}' does not exist.")
        exit()
for idx, filename in enumerate(sorted(os.listdir(folder_path))):
             old_path = os.path.join(folder_path, filename)

             if not os.path.isfile(old_path):
                        continue
             file_extension = os.path.splitext(filename)[1]
             new_filename = f'NT_image_{idx+1:03d}{file_extension}'
             new_path= os.path.join(folder_path, new_filename)

             os.rename(old_path, new_path)
             print(f'Renamed: {filename} -> {new_filename}')

print("completion of file renaming 0_0")           