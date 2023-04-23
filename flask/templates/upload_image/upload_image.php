<!DOCTYPE html>
<html>
<head>
	<title>Upload Image/Video</title>
</head>
<body>
	<form action="process.php" method="post" enctype="multipart/form-data">
		<label for="file">Choose an image :</label>
		<input type="file" name="file" id="file">
		<br>
		<input type="submit" name="submit" value="Upload and Process">
	</form>
</body>
</html>

<?php
// Check if a file was uploaded
if(isset($_FILES['file']) && $_FILES['file']['error'] == 0) {
    $allowed_extensions = array('jpg', 'jpeg', 'png', 'gif');
    $file_extension = pathinfo($_FILES['file']['name'], PATHINFO_EXTENSION);

    // Check if the uploaded file has an allowed extension
    if(in_array($file_extension, $allowed_extensions)) {
        // Move the uploaded file to a folder for processing
        $target_path = 'uploads/' . basename($_FILES['file']['name']);
        move_uploaded_file($_FILES['file']['tmp_name'], $target_path);

        // Pass the file path to a Python script for processing
        $python_script = 'templates\upload_image\upload_image.py';
        $command = escapeshellcmd('python ' . $python_script . ' ' . $target_path);
        $output = shell_exec($command);

        // Display the output from the Python script
        echo $output;
    } else {
        echo 'Invalid file type. Only JPG, JPEG, PNG and GIF files are allowed.';
    }
} else {
    echo 'An error occurred while uploading the file.';
}
?>
