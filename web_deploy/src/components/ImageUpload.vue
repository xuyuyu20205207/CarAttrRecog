<!--
 * @Description: 
 * @Author: Charles
 * @Date: 2021-06-27 13:48:56
 * @LastEditTime: 2021-06-27 21:46:07
 * @LastEditors: Charles
 * @Reference: 
-->
<template>
  <div style="display: inline-block">
    <a-upload
      name="file"
      :multiple="true"
      :show-upload-list="false"
      :headers="headers"
      @change="uploadProfile"
    >
      <a-button style="margin-bottom: 30px">
        <upload-outlined></upload-outlined>
        上传图片
      </a-button>
    </a-upload>
    <br />
    <div class="img1">
        <img :src="currURL1"/>
    </div>
  </div>
  <div style="display: inline-block; float: right">
    <!-- <a-upload
      name="file"
      :multiple="true"
      :show-upload-list="false"
      :headers="headers"
      @change="startDetect"
    > -->
      <a-button style="margin-bottom: 30px" @click="startDetect()">
        <!-- <upload-outlined></upload-outlined> -->
        开始识别
      </a-button>
    <!-- </a-upload> -->
    <br />
    <img :src="currURL2" />
  </div>
</template>
<script lang="ts">
import { message } from "ant-design-vue";
import { UploadOutlined } from "@ant-design/icons-vue";
import { defineComponent, ref } from "vue";
import axios from "axios";

export default defineComponent({
  components: {
    UploadOutlined,
  },
  setup() {
    const currURL1 = ref<string>("");
    const currURL2 = ref<string>("");
    // convert to base64
    const uploadProfile = (fileObj, fileList, event) => {
      console.log(fileObj.file);
      const reader = new FileReader();
      reader.onloadend = function () {
        console.log("RESULT", reader.result);
        currURL1.value = reader.result as string;
      };
      reader.readAsDataURL(fileObj.file.originFileObj);
    };
    // 把uploadProfile改成api axios call
    const startDetect = () => {
      axios.post('http://127.0.0.1:5000/detect', {
                img: currURL1.value
            }).then((reponse) => {
                console.log(reponse);
                currURL2.value = reponse.data as string;
                console.log(currURL2.value);
            })
    };


    return {
      uploadProfile,
      startDetect,
      currURL1,
      currURL2,
    };
  },
});
</script>
<style scoped>
.img1{
  width: 300px;
  height: 200px;
}
.img1{
  width: 70%;
  height: 70%;
}
</style>
