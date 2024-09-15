<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cập nhật phẩm mới</title>
    <!-- CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css" 
    integrity="sha384-xOolHFLEh07PJGoPkLv1IbcEPTNtaed2xpHsD9ESMhqIYd0nLMwNLD69Npy4HI+N" 
    crossorigin="anonymous">
    <!-- JS -->
    <script src="https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.slim.min.js" 
    integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" 
    crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/js/bootstrap.bundle.min.js" 
    integrity="sha384-Fy6S3B9q64WdZWQUiU+q4/2Lc9npb8tCaSX9FK7E8HnRr0Jz8D6OP9dO5Vg3Q9ct" 
    crossorigin="anonymous"></script>
</head>
<body>
    <div class="container">
        <div class="row justify-content-around">
            <form action="" class="col-md-6
            bg-light
            p-3 my-3" 
            method="post"> <!--tối thiểu 6 cột-->
                <h1 class="text-center text-uppercase h3 py-3">Cập nhật sản phẩm</h1>
                <div class="form-group">
                    <label for="txtMaSPCU">Cập nhật dữ liệu mới theo mã sản phẩm cũ:</label>
                    <input type="text" name="txtMaSPCU" id="txtMaSPCU" class="form-control" required>
                </div>
                <div class="form-group">
                    <label for="txtMaSPMOI">Mã sản phẩm mới:</label>
                    <input type="text" name="txtMaSPMOI" id="txtMaSPMOI" class="form-control" required>
                </div>

                <div class="form-group">
                    <label for="txtTenSP">Tên sản phẩm mới:</label>
                    <input type="text" name="txtTenSP" id="txtTenSP" class="form-control" required>
                </div>

                <div class="form-group">
                    <label for="txtGiaSP">Giá mới:</label>
                    <input type="text" name="txtGiaSP" id="txtGiaSP" class="form-control" required>
                </div>
                <input type="submit" name="capnhat" value="Cập nhật" class="btn-primary btn btn-block">
            </form>
        </div>         
    </div>

    <?php
    include_once('db_config.php');
    if(isset($_POST['capnhat']) != null)
    {
        $maspcu = $_POST['txtMaSPCU'];
        $maspmoi = $_POST['txtMaSPMOI'];
        $tensp = $_POST['txtTenSP'];
        $giasp = $_POST['txtGiaSP'];
        if ($maspcu != null){
            $sql = "UPDATE sanpham 
                    SET masp = '$maspmoi',
                    tensp = '$tensp',
                    gia = '$giasp'
                    WHERE masp = '$maspcu';";
            $cnn = new mysqli($servername, $username, $password, $dbname);
            if($cnn->connect_error){
                die('Không thể kết nối đến CSDL' . $dbname);
            }
            else{
                $cnn->query($sql);
                $cnn->close();
                echo '<center><h3>'. $sql . '</h3></center>' . '<br>';
                echo '<center><h3> Cập nhật thành công </h3></center>';
            }       
        }
    }
?>
    <br>
    <div class="text-center ">
        <a href="adminhome.php"><button type="button" class="btn btn-success py-3 px-5"><h3>Home</h3></button></a>
    </div>
</body>
</html>