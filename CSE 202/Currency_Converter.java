/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/GUIForms/JFrame.java to edit this template
 */
package com.mycompany.mavenproject2;

import java.awt.event.KeyEvent;

/**
 *
 * @author Tanvir Ahmed
 */
public class Currency_Converter extends javax.swing.JFrame {

    /**
     * Creates new form Currency_Converter
     */
    public Currency_Converter() {
        initComponents();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        buttonGroup1 = new javax.swing.ButtonGroup();
        buttonGroup2 = new javax.swing.ButtonGroup();
        jPanel1 = new javax.swing.JPanel();
        jLabel1 = new javax.swing.JLabel();
        jTextField1 = new javax.swing.JTextField();
        jLabel2 = new javax.swing.JLabel();
        jTextField2 = new javax.swing.JTextField();
        jRadioButton1 = new javax.swing.JRadioButton();
        jRadioButton2 = new javax.swing.JRadioButton();
        jRadioButton3 = new javax.swing.JRadioButton();
        jRadioButton4 = new javax.swing.JRadioButton();
        jRadioButton5 = new javax.swing.JRadioButton();
        jRadioButton6 = new javax.swing.JRadioButton();
        jButton1 = new javax.swing.JButton();
        jLabel3 = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jLabel1.setText("Input:");

        jTextField1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jTextField1ActionPerformed(evt);
            }
        });
        jTextField1.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                jTextField1KeyPressed(evt);
            }
        });

        jLabel2.setText("Output:");

        buttonGroup1.add(jRadioButton1);
        jRadioButton1.setText("$");
        jRadioButton1.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                jRadioButton1KeyPressed(evt);
            }
        });

        buttonGroup1.add(jRadioButton2);
        jRadioButton2.setText("₹");
        jRadioButton2.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                jRadioButton2KeyPressed(evt);
            }
        });

        buttonGroup1.add(jRadioButton3);
        jRadioButton3.setText("৳");
        jRadioButton3.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                jRadioButton3KeyPressed(evt);
            }
        });

        buttonGroup2.add(jRadioButton4);
        jRadioButton4.setText("$");
        jRadioButton4.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                jRadioButton4KeyPressed(evt);
            }
        });

        buttonGroup2.add(jRadioButton5);
        jRadioButton5.setText("₹");
        jRadioButton5.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                jRadioButton5KeyPressed(evt);
            }
        });

        buttonGroup2.add(jRadioButton6);
        jRadioButton6.setText("৳");
        jRadioButton6.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                jRadioButton6KeyPressed(evt);
            }
        });

        jButton1.setForeground(new java.awt.Color(51, 51, 51));
        jButton1.setText("Convert");
        jButton1.setCursor(new java.awt.Cursor(java.awt.Cursor.HAND_CURSOR));
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });
        jButton1.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                jButton1KeyPressed(evt);
            }
        });

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addGap(18, 18, 18)
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(jPanel1Layout.createSequentialGroup()
                                .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 37, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(jTextField1, javax.swing.GroupLayout.PREFERRED_SIZE, 71, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addComponent(jRadioButton1, javax.swing.GroupLayout.PREFERRED_SIZE, 98, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jRadioButton2, javax.swing.GroupLayout.PREFERRED_SIZE, 98, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jRadioButton3, javax.swing.GroupLayout.PREFERRED_SIZE, 98, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(jPanel1Layout.createSequentialGroup()
                                .addGap(60, 60, 60)
                                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(jRadioButton6, javax.swing.GroupLayout.PREFERRED_SIZE, 98, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(jRadioButton5, javax.swing.GroupLayout.PREFERRED_SIZE, 98, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(jRadioButton4, javax.swing.GroupLayout.PREFERRED_SIZE, 98, javax.swing.GroupLayout.PREFERRED_SIZE)))
                            .addGroup(jPanel1Layout.createSequentialGroup()
                                .addGap(31, 31, 31)
                                .addComponent(jLabel2)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(jTextField2, javax.swing.GroupLayout.PREFERRED_SIZE, 71, javax.swing.GroupLayout.PREFERRED_SIZE))))
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addGap(102, 102, 102)
                        .addComponent(jButton1)))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel1)
                    .addComponent(jTextField1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel2)
                    .addComponent(jTextField2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jRadioButton1)
                    .addComponent(jRadioButton4))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jRadioButton2)
                    .addComponent(jRadioButton5))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jRadioButton3)
                    .addComponent(jRadioButton6))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jButton1)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        jLabel3.setIcon(new javax.swing.ImageIcon("C:\\Users\\Tanvir Ahmed\\Downloads\\1.png")); // NOI18N

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
            .addGroup(layout.createSequentialGroup()
                .addGap(25, 25, 25)
                .addComponent(jLabel3)
                .addGap(0, 0, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jLabel3)
                .addGap(18, 18, 18))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jTextField1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jTextField1ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_jTextField1ActionPerformed

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        double in = Double.parseDouble(jTextField1.getText());
        if(jRadioButton1.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton6.isSelected())
        {
            double out = in * 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton4.isSelected())
        {
            double out = in / 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton6.isSelected())
        {
            double out = in / 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton1.isSelected())
        {
            double out = in / 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton4.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton5.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton6.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }

    }//GEN-LAST:event_jButton1ActionPerformed

    private void jButton1KeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_jButton1KeyPressed

    }//GEN-LAST:event_jButton1KeyPressed

    private void jTextField1KeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_jTextField1KeyPressed
        // TODO add your handling code here:
        if(evt.getKeyCode()== KeyEvent.VK_ENTER){
            double in = Double.parseDouble(jTextField1.getText());
        if(jRadioButton1.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton6.isSelected())
        {
            double out = in * 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton4.isSelected())
        {
            double out = in / 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton6.isSelected())
        {
            double out = in / 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton1.isSelected())
        {
            double out = in / 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton4.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton5.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton6.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        }
    }//GEN-LAST:event_jTextField1KeyPressed

    private void jRadioButton1KeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_jRadioButton1KeyPressed
        // TODO add your handling code here:
        if(evt.getKeyCode()== KeyEvent.VK_ENTER){
            double in = Double.parseDouble(jTextField1.getText());
        if(jRadioButton1.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton6.isSelected())
        {
            double out = in * 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton4.isSelected())
        {
            double out = in / 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton6.isSelected())
        {
            double out = in / 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton1.isSelected())
        {
            double out = in / 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton4.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton5.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton6.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        }
    }//GEN-LAST:event_jRadioButton1KeyPressed

    private void jRadioButton2KeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_jRadioButton2KeyPressed
        // TODO add your handling code here:
        if(evt.getKeyCode()== KeyEvent.VK_ENTER){
            double in = Double.parseDouble(jTextField1.getText());
        if(jRadioButton1.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton6.isSelected())
        {
            double out = in * 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton4.isSelected())
        {
            double out = in / 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton6.isSelected())
        {
            double out = in / 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton1.isSelected())
        {
            double out = in / 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton4.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton5.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton6.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        }
    }//GEN-LAST:event_jRadioButton2KeyPressed

    private void jRadioButton3KeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_jRadioButton3KeyPressed
        // TODO add your handling code here:
        if(evt.getKeyCode()== KeyEvent.VK_ENTER){
            double in = Double.parseDouble(jTextField1.getText());
        if(jRadioButton1.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton6.isSelected())
        {
            double out = in * 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton4.isSelected())
        {
            double out = in / 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton6.isSelected())
        {
            double out = in / 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton1.isSelected())
        {
            double out = in / 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton4.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton5.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton6.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        }
    }//GEN-LAST:event_jRadioButton3KeyPressed

    private void jRadioButton4KeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_jRadioButton4KeyPressed
        // TODO add your handling code here:
        if(evt.getKeyCode()== KeyEvent.VK_ENTER){
            double in = Double.parseDouble(jTextField1.getText());
        if(jRadioButton1.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton6.isSelected())
        {
            double out = in * 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton4.isSelected())
        {
            double out = in / 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton6.isSelected())
        {
            double out = in / 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton1.isSelected())
        {
            double out = in / 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton4.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton5.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton6.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        }
    }//GEN-LAST:event_jRadioButton4KeyPressed

    private void jRadioButton5KeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_jRadioButton5KeyPressed
        // TODO add your handling code here:
        if(evt.getKeyCode()== KeyEvent.VK_ENTER){
            double in = Double.parseDouble(jTextField1.getText());
        if(jRadioButton1.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton6.isSelected())
        {
            double out = in * 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton4.isSelected())
        {
            double out = in / 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton6.isSelected())
        {
            double out = in / 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton1.isSelected())
        {
            double out = in / 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton4.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton5.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton6.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        }
    }//GEN-LAST:event_jRadioButton5KeyPressed

    private void jRadioButton6KeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_jRadioButton6KeyPressed
        // TODO add your handling code here:
        if(evt.getKeyCode()== KeyEvent.VK_ENTER){
            double in = Double.parseDouble(jTextField1.getText());
        if(jRadioButton1.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton6.isSelected())
        {
            double out = in * 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton4.isSelected())
        {
            double out = in / 81.99;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton6.isSelected())
        {
            double out = in / 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton1.isSelected())
        {
            double out = in / 107.01;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton5.isSelected())
        {
            double out = in * 0.77;
            String x = "" + out;
            jTextField2.setText(x);
        }
        if(jRadioButton1.isSelected() && jRadioButton4.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton2.isSelected() && jRadioButton5.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        if(jRadioButton3.isSelected() && jRadioButton6.isSelected())
        {
            String x = "" + in;
            jTextField2.setText(x);
        }
        }
    }//GEN-LAST:event_jRadioButton6KeyPressed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(Currency_Converter.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(Currency_Converter.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(Currency_Converter.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(Currency_Converter.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            @Override
            public void run() {
                new Currency_Converter().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.ButtonGroup buttonGroup1;
    private javax.swing.ButtonGroup buttonGroup2;
    private javax.swing.JButton jButton1;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JRadioButton jRadioButton1;
    private javax.swing.JRadioButton jRadioButton2;
    private javax.swing.JRadioButton jRadioButton3;
    private javax.swing.JRadioButton jRadioButton4;
    private javax.swing.JRadioButton jRadioButton5;
    private javax.swing.JRadioButton jRadioButton6;
    private javax.swing.JTextField jTextField1;
    private javax.swing.JTextField jTextField2;
    // End of variables declaration//GEN-END:variables
}