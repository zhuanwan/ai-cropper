class DocumentScanner {
  constructor() {
    this.video = document.getElementById("camera");
    this.canvas = document.getElementById("photoCanvas");
    this.captureBtn = document.getElementById("captureBtn");
    this.saveBtn = document.getElementById("saveBtn");
    this.preview = document.getElementById("preview");
    this.stream = null;
    this.captures = [];
    this.fileInput = document.getElementById("fileInput");
    this.uploadBtn = document.getElementById("uploadBtn");
    this.confirmBtn = document.getElementById("confirmBtn");
    this.currentEditor = null;
    this.imageHistory = []; // 存储图片历史
    this.currentImageIndex = -1; // 当前图片索引
    this.downloadBtn = document.getElementById("downloadBtn");
    this.imageHistoryContainer = document.getElementById("imageHistory");
    this.savedImages = []; // 存储已保存的图片
    this.editingImageIndex = undefined; // 添加编辑图片索引
    this.isCameraActive = true; // 添加相机状态标记

    this.confirmBtn.onclick = () => this.applyPerspectiveCorrection();

    this.initializeCamera().catch(() => {
      // 如果相机初始化失败，隐藏相机相关元素
      this.video.classList.add("hidden");
      this.captureBtn.style.display = "none";
    });
    this.bindEvents();
  }

  async initializeCamera() {
    try {
      // 检查是否支持 getUserMedia
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("浏览器不支持访问相机");
      }

      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
        },
      });
      this.video.srcObject = this.stream;
    } catch (err) {
      console.error("相机访问失败:", err);
      if (err.name === "NotAllowedError") {
        alert("无法访问相机：权限被拒绝。请在浏览器设置中允许访问相机。");
      } else if (err.name === "NotFoundError") {
        alert("未检测到相机设备，您可以通过'上传图片'功能继续使用。");
      } else {
        alert("相机访问失败：" + err.message + "\n您可以通过'上传图片'功能继续使用。");
      }
      throw err;
    }
  }

  bindEvents() {
    this.captureBtn.addEventListener("click", () => this.captureImage());
    this.saveBtn.addEventListener("click", () => this.saveImage());
    this.uploadBtn.addEventListener("click", () => this.fileInput.click());
    this.fileInput.addEventListener("change", (e) => this.handleFileUpload(e));
    this.downloadBtn.addEventListener("click", () => this.saveToPDF());
  }

  async processImage(imgData) {
    try {
      // 检查输入数据的有效性
      if (!imgData || !imgData.width || !imgData.height) {
        console.error("无效的图像数据");
        return null;
      }

      // 将 ImageData 转换为 Mat 对象
      let src = cv.matFromImageData(imgData);
      if (src.empty()) {
        console.error("图像转换失败");
        return null;
      }

      let dst = new cv.Mat();
      let edges = new cv.Mat();
      let hierarchy = new cv.Mat();
      let contours = new cv.MatVector();

      // 转换为灰度图
      cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);

      // 高斯模糊减少噪声
      cv.GaussianBlur(dst, dst, new cv.Size(5, 5), 0);

      // Canny 边缘检测
      cv.Canny(dst, edges, 75, 200);

      // 膨胀边缘
      let kernel = cv.Mat.ones(5, 5, cv.CV_8U);
      cv.dilate(edges, edges, kernel, new cv.Point(-1, -1), 1);

      // 查找轮廓
      cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      // 查找最大的矩形轮廓
      let maxArea = 0;
      let maxContourIndex = -1;
      let documentCorners = null;

      for (let i = 0; i < contours.size(); ++i) {
        let contour = contours.get(i);
        let area = cv.contourArea(contour);
        let perimeter = cv.arcLength(contour, true);
        let approx = new cv.Mat();
        cv.approxPolyDP(contour, approx, 0.02 * perimeter, true);

        // 查找近似四边形的轮廓
        if (area > maxArea && approx.rows === 4) {
          maxArea = area;
          maxContourIndex = i;
          documentCorners = approx;
        }

        if (i !== maxContourIndex) {
          approx.delete();
        }
      }

      // 如果找到文档边界，进行透视变换
      if (documentCorners) {
        // 对角点排序
        let corners = this.sortCorners(documentCorners);

        // 计算文档尺寸
        let widthBottom = Math.sqrt(
          Math.pow(corners[2].x - corners[3].x, 2) + Math.pow(corners[2].y - corners[3].y, 2)
        );
        let widthTop = Math.sqrt(
          Math.pow(corners[1].x - corners[0].x, 2) + Math.pow(corners[1].y - corners[0].y, 2)
        );
        let width = Math.max(widthTop, widthBottom);

        let heightRight = Math.sqrt(
          Math.pow(corners[1].x - corners[2].x, 2) + Math.pow(corners[1].y - corners[2].y, 2)
        );
        let heightLeft = Math.sqrt(
          Math.pow(corners[0].x - corners[3].x, 2) + Math.pow(corners[0].y - corners[3].y, 2)
        );
        let height = Math.max(heightRight, heightLeft);

        // 创建透视变换的源点和目标点
        let srcPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
          corners[0].x,
          corners[0].y,
          corners[1].x,
          corners[1].y,
          corners[2].x,
          corners[2].y,
          corners[3].x,
          corners[3].y,
        ]);

        let dstPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
          0,
          0,
          width - 1,
          0,
          width - 1,
          height - 1,
          0,
          height - 1,
        ]);

        // 计算透视变换矩阵
        let perspectiveMatrix = cv.getPerspectiveTransform(srcPoints, dstPoints);

        // 应用透视变换
        let warped = new cv.Mat();
        cv.warpPerspective(src, warped, perspectiveMatrix, new cv.Size(width, height));

        // 检查透视变换后的图像是否有效
        if (warped.empty() || warped.cols === 0 || warped.rows === 0) {
          console.error("透视变换失败");
          warped.delete();
          return null;
        }

        // 确保图像数据是有效的
        if (warped.data.length !== warped.cols * warped.rows * warped.channels()) {
          console.error("图像数据不完整");
          warped.delete();
          return null;
        }

        // 清理内存
        src.delete();
        dst.delete();
        edges.delete();
        hierarchy.delete();
        contours.delete();
        documentCorners.delete();
        kernel.delete();
        srcPoints.delete();
        dstPoints.delete();
        perspectiveMatrix.delete();

        // 返回处理后的图像
        return warped;
      }

      // 清理内存
      src.delete();
      dst.delete();
      edges.delete();
      hierarchy.delete();
      contours.delete();
      kernel.delete();

      return null;
    } catch (err) {
      console.error("图像处理失败:", err);
      return null;
    }
  }

  sortCorners(corners) {
    let pts = [];
    for (let i = 0; i < corners.rows; i++) {
      pts.push({
        x: corners.data32S[i * 2],
        y: corners.data32S[i * 2 + 1],
      });
    }

    // 计算质心
    let center = pts.reduce(
      (acc, pt) => {
        return { x: acc.x + pt.x / 4, y: acc.y + pt.y / 4 };
      },
      { x: 0, y: 0 }
    );

    // 将点分为左上、右上、右下、左下
    return pts.sort((a, b) => {
      let a1 = Math.atan2(a.y - center.y, a.x - center.x);
      let a2 = Math.atan2(b.y - center.y, b.x - center.x);
      return a1 - a2;
    });
  }

  async captureImage() {
    if (!this.isCameraActive) return;

    try {
      const context = this.canvas.getContext("2d");
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;

      if (!this.canvas.width || !this.canvas.height) {
        throw new Error("无效的视频尺寸");
      }

      context.drawImage(this.video, 0, 0);

      // 获取图像数据
      const imgData = context.getImageData(0, 0, this.canvas.width, this.canvas.height);

      // 创建图片元素并等待加载
      const img = new Image();
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = this.canvas.toDataURL("image/jpeg");
      });

      // 清除预览区域
      this.preview.innerHTML = "";

      // 创建编辑器
      this.createDocumentEditor(img);

      // 显示确认按钮
      this.confirmBtn.style.display = "inline-block";

      // 禁用拍摄按钮
      this.captureBtn.disabled = true;
      this.isCameraActive = false;

      // 添加到历史记录
      this.imageHistory.push({
        src: img.src,
        isProcessed: false,
      });
      this.currentImageIndex = this.imageHistory.length - 1;
    } catch (err) {
      console.error("拍摄失败:", err);
      alert("拍摄失败，请重试");
    }
  }

  async saveImage() {
    if (!this.currentEditor) return;

    const imgElement = this.currentEditor.querySelector("img");
    if (!imgElement) return;

    try {
      // 创建临时 canvas 以更高质量保存图片
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = imgElement.naturalWidth;
      tempCanvas.height = imgElement.naturalHeight;
      const ctx = tempCanvas.getContext("2d");

      // 使用更好的图像渲染设置
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(imgElement, 0, 0);

      // 使用更高质量的图片格式和设置
      const highQualityImageUrl = tempCanvas.toDataURL("image/jpeg", 1.0);

      if (this.editingImageIndex !== undefined) {
        this.savedImages.splice(this.editingImageIndex, 1);
        this.editingImageIndex = undefined;
      }

      // 创建历史记录项
      const historyItem = document.createElement("div");
      historyItem.className = "history-item";

      const img = document.createElement("img");
      img.src = highQualityImageUrl; // 使用高质量图片

      // 创建删除按钮
      const deleteBtn = document.createElement("div");
      deleteBtn.className = "delete-btn";
      deleteBtn.textContent = "×";
      deleteBtn.addEventListener("click", (e) => {
        e.stopPropagation(); // 阻止事件冒泡
        const index = this.savedImages.indexOf(img.src);
        if (index > -1) {
          this.savedImages.splice(index, 1);
          historyItem.remove();
          this.downloadBtn.disabled = this.savedImages.length === 0;
        }
      });

      // 添加点击事件
      historyItem.addEventListener("click", () => {
        this.editingImageIndex = this.savedImages.indexOf(img.src);
        this.editHistoryImage(img.src);
      });

      // 组装历史记录项
      historyItem.appendChild(deleteBtn);
      historyItem.appendChild(img);
      this.imageHistoryContainer.appendChild(historyItem);

      // 保存高质量图片
      this.savedImages.push(highQualityImageUrl);

      // 更新下载按钮状态
      this.downloadBtn.disabled = false;

      // 清理编辑器和预览
      this.currentEditor.remove();
      this.currentEditor = null;
      this.preview.innerHTML = "";
      this.saveBtn.disabled = true;

      // 在保存完成后启用拍摄按钮
      this.captureBtn.disabled = false;
      this.isCameraActive = true;
    } catch (err) {
      console.error("保存失败:", err);
      alert("保存失败，请重试");
    }
  }

  async editHistoryImage(imageSrc) {
    try {
      // 创建新的图片对象
      const img = new Image();

      // 使用 Promise 等待图片加载完成
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = imageSrc;
      });

      // 清除预览区域
      this.preview.innerHTML = "";

      // 重置按钮状态
      this.confirmBtn.style.display = "inline-block";
      this.saveBtn.disabled = true;
      this.captureBtn.disabled = false;
      this.uploadBtn.disabled = false;

      // 创建一个临时 canvas 来处理图片数据
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = img.naturalWidth;
      tempCanvas.height = img.naturalHeight;
      const ctx = tempCanvas.getContext("2d");
      ctx.drawImage(img, 0, 0);

      // 获取新的图片数据
      const newImgSrc = tempCanvas.toDataURL("image/jpeg", 1.0);
      const newImg = new Image();
      newImg.src = newImgSrc;

      // 创建编辑器
      this.createDocumentEditor(newImg);
    } catch (err) {
      console.error("编辑历史图片失败:", err);
      alert("加载图片失败，请重试");
    }
  }

  async saveToPDF() {
    if (this.savedImages.length === 0) {
      alert("没有可下载的图片");
      return;
    }

    try {
      const { jsPDF } = window.jspdf;
      // 创建 A4 大小的 PDF，使用更高质量的设置
      const doc = new jsPDF({
        orientation: "portrait",
        unit: "mm",
        format: "a4",
        compress: false, // 禁用压缩以保持图片质量
      });

      const pageWidth = doc.internal.pageSize.getWidth();
      const pageHeight = doc.internal.pageSize.getHeight();
      const margin = 10;

      for (let i = 0; i < this.savedImages.length; i++) {
        if (i > 0) doc.addPage();

        // 加载图片并等待
        const tempImg = await this.loadImage(this.savedImages[i]);

        // 计算图片的原始宽高比
        const imgRatio = tempImg.width / tempImg.height;

        // 计算可用空间
        const availableWidth = pageWidth - 2 * margin;
        const availableHeight = pageHeight - 2 * margin;

        // 计算最佳适配尺寸
        let finalWidth, finalHeight;
        if (imgRatio > availableWidth / availableHeight) {
          finalWidth = availableWidth;
          finalHeight = finalWidth / imgRatio;
        } else {
          finalHeight = availableHeight;
          finalWidth = finalHeight * imgRatio;
        }

        // 计算居中位置
        const x = margin + (availableWidth - finalWidth) / 2;
        const y = margin + (availableHeight - finalHeight) / 2;

        // 添加图片到 PDF，使用更高的图片质量设置
        doc.addImage(
          this.savedImages[i],
          "JPEG",
          x,
          y,
          finalWidth,
          finalHeight,
          undefined,
          "FAST",
          0
        );
      }

      // 保存 PDF
      doc.save("scanned_document.pdf");
    } catch (err) {
      console.error("PDF生成失败:", err);
      alert("PDF生成失败，请重试");
    }
  }

  async handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    try {
      const img = new Image();
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
      });

      // 清除预览区域
      this.preview.innerHTML = "";

      // 创建编辑器
      this.createDocumentEditor(img);

      // 显示确认按钮
      this.confirmBtn.style.display = "inline-block";

      // 添加到历史记录
      this.imageHistory.push({
        src: img.src,
        isProcessed: false,
      });
      this.currentImageIndex = this.imageHistory.length - 1;
    } catch (err) {
      console.error("文件加载失败:", err);
      alert("文件加载失败，请重试");
    }

    // 清除文件输入，允许重复选择相同文件
    this.fileInput.value = "";
  }

  loadImage(source) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      // 如果是文件对象，创建 URL
      if (source instanceof File) {
        img.src = URL.createObjectURL(source);
      } else {
        // 如果是 URL 字符串，直接使用
        img.src = source;
      }
    });
  }

  // 添加一个辅助方法来显示原始图像
  useOriginalImage() {
    const img = document.createElement("img");
    img.src = this.canvas.toDataURL("image/jpeg");
    this.preview.appendChild(img);
    this.captures.push(img.src);
    this.saveBtn.disabled = false;
  }

  createDocumentEditor(img, corners = null) {
    // 清除之前的编辑器
    if (this.currentEditor) {
      this.currentEditor.remove();
    }

    const editor = document.createElement("div");
    editor.className = "document-editor";

    // 创建图片容器
    const imgContainer = document.createElement("div");
    imgContainer.style.position = "relative";

    // 添加图片
    const imgElement = document.createElement("img");
    imgElement.src = img.src;
    imgContainer.appendChild(imgElement);

    // 修改图片加载完成后的处理
    imgElement.onload = () => {
      const imageRect = imgElement.getBoundingClientRect();
      const scale = img.naturalWidth / imageRect.width;

      // 如果没有提供角点，创建默认角点
      if (!corners) {
        corners = [
          { x: 0, y: 0 },
          { x: imageRect.width, y: 0 },
          { x: imageRect.width, y: imageRect.height },
          { x: 0, y: imageRect.height },
        ];
      }

      // 创建角点手柄容器
      const handleContainer = document.createElement("div");
      handleContainer.className = "corner-handle-container";

      // 创建四个角落的控制点
      corners.forEach((corner, index) => {
        const handle = document.createElement("div");
        handle.className = "corner-handle";
        handle.style.left = `${corner.x}px`;
        handle.style.top = `${corner.y}px`;
        this.makeHandleDraggable(handle, imgElement, scale);
        handleContainer.appendChild(handle);
      });

      // 创建文档轮廓
      const outline = document.createElement("div");
      outline.className = "document-outline";
      handleContainer.appendChild(outline);

      imgContainer.appendChild(handleContainer);
      this.updateDocumentOutline(handleContainer);

      // 显示确认按钮
      this.confirmBtn.style.display = "inline-block";
    };

    editor.appendChild(imgContainer);
    this.currentEditor = editor;
    this.preview.appendChild(editor);
  }

  makeHandleDraggable(handle, imgElement, scale) {
    let isDragging = false;
    let currentX;
    let currentY;

    // 修改触摸事件处理
    const touchStart = (e) => {
      isDragging = true;
      const touch = e.touches[0];
      const rect = handle.getBoundingClientRect();
      currentX = touch.clientX - rect.left;
      currentY = touch.clientY - rect.top;
      handle.style.backgroundColor = "#45a049";
    };

    const touchMove = (e) => {
      if (!isDragging) return;
      e.preventDefault(); // 在 move 时阻止滚动

      const touch = e.touches[0];
      const imageRect = imgElement.getBoundingClientRect();
      let newX = touch.clientX - currentX - imageRect.left;
      let newY = touch.clientY - currentY - imageRect.top;

      // 限制在图片范围内
      newX = Math.max(0, Math.min(newX, imageRect.width));
      newY = Math.max(0, Math.min(newY, imageRect.height));

      handle.style.left = newX + "px";
      handle.style.top = newY + "px";

      this.updateDocumentOutline(handle.parentElement);
    };

    const touchEnd = () => {
      if (isDragging) {
        handle.style.backgroundColor = "#4caf50";
        isDragging = false;
      }
    };

    // 添加触摸事件监听
    handle.addEventListener("touchstart", touchStart, { passive: true });
    handle.addEventListener("touchmove", touchMove, { passive: false });
    handle.addEventListener("touchend", touchEnd);
    handle.addEventListener("touchcancel", touchEnd);

    // 鼠标事件保持不变
    handle.addEventListener("mousedown", (e) => {
      isDragging = true;
      currentX = e.clientX - handle.offsetLeft;
      currentY = e.clientY - handle.offsetTop;
      handle.style.backgroundColor = "#45a049";
      e.preventDefault();
    });

    document.addEventListener("mousemove", (e) => {
      if (!isDragging) return;

      const imageRect = imgElement.getBoundingClientRect();
      let newX = e.clientX - currentX;
      let newY = e.clientY - currentY;

      newX = Math.max(0, Math.min(newX, imageRect.width));
      newY = Math.max(0, Math.min(newY, imageRect.height));

      handle.style.left = newX + "px";
      handle.style.top = newY + "px";

      this.updateDocumentOutline(handle.parentElement);
    });

    document.addEventListener("mouseup", () => {
      if (isDragging) {
        handle.style.backgroundColor = "#4caf50";
        isDragging = false;
      }
    });
  }

  updateDocumentOutline(container) {
    const handles = Array.from(container.getElementsByClassName("corner-handle"));
    const outline = container.querySelector(".document-outline");

    if (handles.length === 4) {
      // 清除现有的线条
      outline.innerHTML = "";

      // 创建四条线连接角点
      for (let i = 0; i < 4; i++) {
        const start = handles[i];
        const end = handles[(i + 1) % 4];

        const line = document.createElement("div");
        line.className = "outline-line";

        // 计算线条位置和旋转
        const length = Math.sqrt(
          Math.pow(end.offsetLeft - start.offsetLeft, 2) +
            Math.pow(end.offsetTop - start.offsetTop, 2)
        );

        const angle = Math.atan2(
          end.offsetTop - start.offsetTop,
          end.offsetLeft - start.offsetLeft
        );

        // 设置线条样式
        line.style.width = `${length}px`;
        line.style.left = `${start.offsetLeft}px`;
        line.style.top = `${start.offsetTop}px`;
        line.style.transform = `rotate(${angle}rad)`;

        outline.appendChild(line);
      }
    }
  }

  async applyPerspectiveCorrection() {
    if (!this.currentEditor) return;

    const imgElement = this.currentEditor.querySelector("img");
    const handles = Array.from(this.currentEditor.getElementsByClassName("corner-handle"));

    if (!imgElement || handles.length !== 4) return;

    const scale = imgElement.naturalWidth / imgElement.getBoundingClientRect().width;
    const corners = handles.map((handle) => ({
      x: handle.offsetLeft * scale,
      y: handle.offsetTop * scale,
    }));

    try {
      // 将图片绘制到canvas
      this.canvas.width = imgElement.naturalWidth;
      this.canvas.height = imgElement.naturalHeight;
      const context = this.canvas.getContext("2d");
      context.drawImage(imgElement, 0, 0);

      // 应用透视变换
      const processedMat = await this.applyTransform(
        context.getImageData(0, 0, this.canvas.width, this.canvas.height),
        corners
      );

      if (processedMat) {
        try {
          // 转换Mat为ImageData
          let processedImgData = new ImageData(
            new Uint8ClampedArray(processedMat.data),
            processedMat.cols,
            processedMat.rows
          );

          // 创建canvas并绘制图像
          const canvas = document.createElement("canvas");
          canvas.width = processedMat.cols;
          canvas.height = processedMat.rows;
          const ctx = canvas.getContext("2d");
          ctx.putImageData(processedImgData, 0, 0);

          // 获取图片URL
          const imgUrl = canvas.toDataURL("image/jpeg");

          // 清除当前编辑器
          this.currentEditor.remove();

          // 创建新的编辑器实例
          const newImg = new Image();
          newImg.onload = () => {
            this.createDocumentEditor(newImg);
          };
          newImg.src = imgUrl;

          // 启用保存按钮
          this.saveBtn.disabled = false;

          // 清理内存
          processedMat.delete();
        } catch (err) {
          console.error("透视变换失败:", err);
          alert("透视变换失败，请重试");
        }
      }
    } catch (err) {
      console.error("处理失败:", err);
      alert("处理失败，请重试");
    }
  }

  async applyTransform(imgData, corners) {
    try {
      let src = cv.matFromImageData(imgData);

      // 计算目标尺寸
      const width = Math.max(
        Math.sqrt(
          Math.pow(corners[1].x - corners[0].x, 2) + Math.pow(corners[1].y - corners[0].y, 2)
        ),
        Math.sqrt(
          Math.pow(corners[2].x - corners[3].x, 2) + Math.pow(corners[2].y - corners[3].y, 2)
        )
      );

      const height = Math.max(
        Math.sqrt(
          Math.pow(corners[3].x - corners[0].x, 2) + Math.pow(corners[3].y - corners[0].y, 2)
        ),
        Math.sqrt(
          Math.pow(corners[2].x - corners[1].x, 2) + Math.pow(corners[2].y - corners[1].y, 2)
        )
      );

      // 设置变换矩阵
      let srcPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
        corners[0].x,
        corners[0].y,
        corners[1].x,
        corners[1].y,
        corners[2].x,
        corners[2].y,
        corners[3].x,
        corners[3].y,
      ]);

      let dstPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0,
        0,
        width - 1,
        0,
        width - 1,
        height - 1,
        0,
        height - 1,
      ]);

      // 应用变换
      let perspectiveMatrix = cv.getPerspectiveTransform(srcPoints, dstPoints);
      let warped = new cv.Mat();
      cv.warpPerspective(src, warped, perspectiveMatrix, new cv.Size(width, height));

      // 清理
      src.delete();
      srcPoints.delete();
      dstPoints.delete();
      perspectiveMatrix.delete();

      return warped;
    } catch (err) {
      console.error("变换失败:", err);
      return null;
    }
  }

  displayProcessedImage(processedMat) {
    // 将处理后的 Mat 转换为 ImageData
    let processedImgData = new ImageData(
      new Uint8ClampedArray(processedMat.data),
      processedMat.cols,
      processedMat.rows
    );

    // 创建新的 canvas 来显示处理后的图像
    const processedCanvas = document.createElement("canvas");
    processedCanvas.width = processedMat.cols;
    processedCanvas.height = processedMat.rows;
    const processedContext = processedCanvas.getContext("2d");
    processedContext.putImageData(processedImgData, 0, 0);

    // 创建预览图像
    const previewImg = document.createElement("img");
    previewImg.src = processedCanvas.toDataURL("image/jpeg");
    this.preview.appendChild(previewImg);
    this.captures.push(previewImg.src);
    this.saveBtn.disabled = false;

    // 添加到历史记录
    this.imageHistory.push({
      src: previewImg.src,
      isProcessed: true,
    });
    this.currentImageIndex = this.imageHistory.length - 1;

    // 显示处理后的图片
    this.showCurrentImage();

    // 清理内存
    processedMat.delete();
  }

  // 显示当前图片
  showCurrentImage() {
    const currentImage = this.imageHistory[this.currentImageIndex];

    // 清除预览区域
    this.preview.innerHTML = "";

    // 默认隐藏确认按钮
    this.confirmBtn.style.display = "none";

    if (currentImage.isProcessed) {
      // 如果是处理后的图片，直接显示
      const img = document.createElement("img");
      img.src = currentImage.src;
      this.preview.appendChild(img);
    } else {
      // 如果是原始图片，创建编辑器
      const img = new Image();
      img.onload = () => {
        this.createDocumentEditor(img, currentImage.corners);
        // 显示确认按钮并绑定事件
        this.confirmBtn.style.display = "inline-block";
        this.confirmBtn.onclick = () => this.applyPerspectiveCorrection();
      };
      img.src = currentImage.src;
    }

    // 更新按钮状态
    this.saveBtn.disabled = !this.hasProcessedImages();
  }

  // 检查是否有处理完成的图片
  hasProcessedImages() {
    return this.imageHistory.some((img) => img.isProcessed);
  }

  // 可选：添加取消编辑的功能
  cancelEdit() {
    if (this.currentEditor) {
      this.currentEditor.remove();
      this.currentEditor = null;
      this.preview.innerHTML = "";
      this.confirmBtn.style.display = "none";
      this.saveBtn.disabled = true;
      // 恢复拍摄功能
      this.captureBtn.disabled = false;
      this.isCameraActive = true;
    }
  }
}

// 等待OpenCV.js加载完成
window.onload = function () {
  if (typeof cv !== "undefined") {
    new DocumentScanner();
  } else {
    // OpenCV.js 加载失败时的处理
    console.error("OpenCV.js 加载失败");
  }
};

function confirmCrop() {
  // ... 现有的裁剪确认代码 ...

  // 在确认裁剪后重置状态
  resetCropState();
}

// 添加一个新的重置状态的函数
function resetCropState() {
  // 重置所有点击事件监听
  const preview = document.getElementById("preview");
  preview.innerHTML = ""; // 清空预览区域

  // 重置按钮状态
  document.getElementById("confirmBtn").style.display = "none";
  document.getElementById("uploadBtn").disabled = false;
  document.getElementById("captureBtn").disabled = false;
  document.getElementById("saveBtn").disabled = false;

  // 重置裁剪点数组
  points = [];
  currentPoint = null;

  // 移除之前的事件监听器
  preview.removeEventListener("click", handlePreviewClick);
  preview.removeEventListener("mousemove", handlePreviewMouseMove);

  // 重新添加事件监听器
  preview.addEventListener("click", handlePreviewClick);
  preview.addEventListener("mousemove", handlePreviewMouseMove);
}

// 在处理新图片时（无论是上传还是拍摄）都调用重置函数
function handleNewImage(imageData) {
  resetCropState();
  // ... 现有的图片处理代码 ...
}
