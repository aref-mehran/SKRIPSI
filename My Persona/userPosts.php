<!DOCTYPE html>
<html>
<head>
</head>
<body>
	<?php
	$DB_HOST = "localhost";
	$DB_USERNAME = "id1693464_mypersona";
	$DB_PASSWORD = "Thinkyoursin3";
	$DB_NAME = "id1693464_mypersona";

	$con = mysqli_connect($DB_HOST, $DB_USERNAME, $DB_PASSWORD, $DB_NAME);
	if (!$con)
	{
		die('Could not connect: ' . mysqli_error($con));
	}

	$insert_id = $_GET['insert_id'];
	$user_id = $_GET['user_id'];
	$user_name = str_replace('"', '', urldecode($_GET['user_name']));
	$status_update = str_replace('"', '', urldecode($_GET['status_update']));

	mysqli_select_db($con, $DB_NAME);
	$query = 'INSERT INTO user_posts(insert_id, user_id, user_name, status_update) VALUES ("'.$insert_id.'", "'.$user_id.'", "'.$user_name.'", "'.$status_update.'");';
	$result = mysqli_query($con, $query);
	
	mysqli_close($con);
	?>
</body>
</html>