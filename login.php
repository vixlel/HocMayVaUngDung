<?php
    $host="localhost";
    $user="root";
    $password="";
    $db="user";
    
    session_start();
    
    
    $data=mysqli_connect($host,$user,$password,$db);
    
    if($data===false)
    {
        die("connection error");
    }
    
    
    if($_SERVER["REQUEST_METHOD"]=="POST")
    {
        $username=$_POST["username"];
        $password=$_POST["password"];
    
    
        $sql="select * from login where username='".$username."' AND password='".$password."' ";
    
        $result=mysqli_query($data,$sql);
    
        $row=mysqli_fetch_array($result);
    
        if($row["usertype"]=="user")
        {	
    
            $_SESSION["username"]=$username;
    
            header("location:userhome.php");
        }
    
        elseif($row["usertype"]=="admin")
        {
    
            $_SESSION["username"]=$username;
            
            header("location:adminhome.php");
        }
    
        else
        {
            echo "username or password incorrect";
        }
    
    }
    
?>





<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log In</title>
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
                <h1 class="text-center text-uppercase h3 py-3">Đăng nhập tài khoản</h1>
                <div class="form-group">
                    <label for="username">Username:</label>
                    <input type="text" name="username" id="username" class="form-control" required>
                </div>
                <div class="form-group">
                    <label for="password">Password:</label>
                    <input type="password" name="password" id="password" class="form-control" required>
                </div>
                <input type="submit" name="dangnhap" value="Log In" class="btn-primary btn btn-block">
            </form>
        </div>         
    </div>
</body>
</html>

