<?php
session_start();


if(!isset($_SESSION["username"]))
{
	header("location:index.php");
}

?>

<!DOCTYPE html>
<html>
<head>
	<title>Admin Page</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Oswald&family=Poppins&family=Roboto&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="styles/header.css">
    <link rel="stylesheet" href="styles/sidebar.css">
    <link rel="stylesheet" href="styles/product.css">
    <link rel="stylesheet" href="styles/quantity.css">
    <style>
        table, th, td {
            font-size: 32px;
            border: 2px solid black;
            border-collapse: collapse;
            font-family: 'Poppins', sans-serif;
        }
        th{
            color:blue;
            background-color: lightgrey;
            padding: 8px 8px 8px 8px;
        }
        td{
            padding: 8px 8px 8px 8px;
        }
        table{
            display: flex;
            flex: 1;
            justify-content: center;
            align-items: center;
            border: none;
        }
        .log-out-btn{
            display: flex;
            flex-direction: row;
            justify-content: end;
            margin-right: 20px;
            margin-top: 20px;
            font-size: 16px;
        }
        .chinhsua{
            font-size: 16px;
            background-color: skyblue;
            font-family: 'Oswald', sans-serif;
        }
    </style>
</head>
<body>
    <a href="logout.php" class ="log-out-btn"><button class="chinhsua">Log out</button></a>
    <?php        
    include_once('db_config.php');
    $cnn = new mysqli($servername, $username, $password, $dbname);
    if($cnn->connect_error){
        die('Không thể kết nối đến CSDL' . $dbname);
    }
    else{
        echo '<center><h1> QUẢN LÝ SẢN PHẨM </h1></center>';
        $truy_van = 'Select * from sanpham';
        $ds_sanpham = $cnn->query($truy_van);

        $table = "<table>";
        $table .= "<th>Mã sản phẩm</th>";
        $table .= "<th>Tên sản phẩm</th>";
        $table .= "<th>Giá sản phẩm</th>";
        while($dong = $ds_sanpham->fetch_assoc()){
            $table .= "<tr>";
            $table .= "<td>" . $dong['masp'] . "</td>";
            $table .= "<td>" . $dong['tensp'] . "</td>";
            $table .= "<td>" . $dong['gia'] . "</td>";
            $table .= "</tr>";
        }

        $table .= "</table>";
        echo $table;
    }
    $cnn->close();
    ?>
    <br><br>
    <div class="btn-container">
        <a href="addnewproduct.php"><button class="interact-btn" type="button">Thêm sản phẩm</button></a>
        <a href="updateproduct.php"><button class="interact-btn" type="button">Cập nhập sản phẩm</button></a>
        <a href="deleteproduct.php"><button class="interact-btn" type="button">Xóa sản phẩm</button></a>
    </div>
</body>
</html>