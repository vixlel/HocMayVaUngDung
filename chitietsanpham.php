<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Danh mục sản phẩm</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Oswald&family=Poppins&family=Roboto&display=swap" rel="stylesheet">
    <!-- CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css" 
    integrity="sha384-xOolHFLEh07PJGoPkLv1IbcEPTNtaed2xpHsD9ESMhqIYd0nLMwNLD69Npy4HI+N" 
    crossorigin="anonymous">
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
    </style>
</head>
<body>
    <?php        
    include_once('db_config.php');
    $cnn = new mysqli($servername, $username, $password, $dbname);
    if($cnn->connect_error){
        die('Không thể kết nối đến CSDL' . $dbname);
    }
    else{
        echo '<center><h1> Kết nối thành công </h1></center>';
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
    <br>
    <div class="text-center ">
        <a href="index.html"><button type="button" class="btn btn-success py-3 px-5"><h3>Home</h3></button></a>
    </div>
</body>
</html>

