USE [bank_projects]
GO
/****** Object:  StoredProcedure [dbo].[spValueCount]    Script Date: 28-01-2025 17:45:27 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
ALTER Procedure [dbo].[spValueCount]
(
@TableName varchar(30),
@ColumnName varchar(30)
)
As 
Begin
Declare @SQLString nvarchar(1000)
Set @SQLString='Select ' + @ColumnName  + ', count(SK_ID_CURR) from ' + @TableName + ' group by ' + @ColumnName

EXEC sp_executesql @SQLString
End